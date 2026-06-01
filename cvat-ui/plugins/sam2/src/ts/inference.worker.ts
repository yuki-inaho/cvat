// Copyright (C) 2024 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { InferenceSession, env, Tensor } from 'onnxruntime-web';

let decoder: InferenceSession | null = null;

env.wasm.wasmPaths = '/assets/';

export enum WorkerAction {
    INIT = 'init',
    DECODE = 'decode',
}

export interface InitBody {
    decoderURL: string;
}

export interface DecodeBody {
    imageEmbed: Float32Array;
    highResFeats0: Float32Array;
    highResFeats1: Float32Array;
    pointCoords: Float32Array;
    pointLabels: Float32Array;
    maskInput: Float32Array | null;
    width: number;
    height: number;
}

export interface WorkerOutput {
    action: WorkerAction;
    error?: string;
}

export interface WorkerInput {
    action: WorkerAction;
    payload: InitBody | DecodeBody;
}

export interface SAM2OutputItem {
    masks: ArrayLike<number>;
    lowResMasks: Float32Array | null;
    bounds: [number, number, number, number];
}

const errorToMessage = (error: unknown): string => {
    if (error instanceof Error) {
        return error.message;
    }
    if (typeof error === 'string') {
        return error;
    }

    console.error(error);
    return 'Unknown error, please check console';
};

// eslint-disable-next-line no-restricted-globals
if ((self as any).importScripts) {
    onmessage = (e: MessageEvent<WorkerInput>) => {
        if (e.data.action === WorkerAction.INIT) {
            if (decoder) {
                return;
            }

            const body = e.data.payload as InitBody;
            InferenceSession.create(body.decoderURL).then((decoderSession) => {
                decoder = decoderSession;
                postMessage({ action: WorkerAction.INIT });
            }).catch((error: unknown) => {
                postMessage({ action: WorkerAction.INIT, error: errorToMessage(error) });
            });
        } else if (!decoder) {
            postMessage({
                action: e.data.action,
                error: 'Worker was not initialized',
            });
        } else if (e.data.action === WorkerAction.DECODE) {
            const body = e.data.payload as DecodeBody;
            const inputs: Record<string, Tensor> = {
                image_embed: new Tensor('float32', body.imageEmbed, [1, 256, 64, 64]),
                high_res_feats_0: new Tensor('float32', body.highResFeats0, [1, 32, 256, 256]),
                high_res_feats_1: new Tensor('float32', body.highResFeats1, [1, 64, 128, 128]),
                point_coords: new Tensor('float32', body.pointCoords, [1, body.pointCoords.length / 2, 2]),
                point_labels: new Tensor('float32', body.pointLabels, [1, body.pointLabels.length]),
                orig_im_size: new Tensor('int32', new Int32Array([body.height, body.width]), [2]),
                mask_input: body.maskInput ?
                    new Tensor('float32', body.maskInput, [1, 1, 256, 256]) :
                    new Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]),
                has_mask_input: new Tensor('float32', new Float32Array([body.maskInput ? 1 : 0]), [1]),
            };

            // Decoder-agnostic output handling. The two supported decoders emit `masks` in
            // different layouts, so we normalise both into the CVAT mask format (a mask cropped
            // to its bounding box plus inclusive [left, top, right, bottom] bounds):
            //   - base_plus: `masks` is multimask logits [1, C, H, W] (C=4) over the FULL image.
            //                We pick the best channel via argmax(iou), binarise (>0), then derive
            //                the bbox and crop the full-image mask to it.
            //   - large:     `masks` is a binary mask [1, 1, H, W] ALREADY cropped to its bbox,
            //                and the absolute bbox is provided via xtl/ytl/xbr/ybr outputs. We use
            //                those bounds directly (the mask is already crop-sized).
            // Reading xtl/ytl/xbr/ybr only when present keeps base_plus support (which lacks them)
            // while preserving large's absolute mask placement.
            decoder.run(inputs).then((results) => {
                const masksT = results.masks;
                const iouT = results.iou_predictions;
                const lowResT = results.low_res_masks; // may be undefined on some decoders
                const { xtl, ytl, xbr, ybr } = results; // large-only absolute bbox outputs
                return Promise.all([
                    masksT.getData(),
                    iouT ? iouT.getData() : Promise.resolve(null),
                    lowResT ? lowResT.getData() : Promise.resolve(null),
                    xtl ? xtl.getData() : Promise.resolve(null),
                    ytl ? ytl.getData() : Promise.resolve(null),
                    xbr ? xbr.getData() : Promise.resolve(null),
                    ybr ? ybr.getData() : Promise.resolve(null),
                ]).then(([maskData, iouData, lowResData, xtlData, ytlData, xbrData, ybrData]) => ({
                    dims: masksT.dims as readonly number[],
                    maskData: maskData as ArrayLike<number>,
                    iouData: iouData as ArrayLike<number> | null,
                    lowResData: lowResData as Float32Array | null,
                    decoderBounds: (xtlData && ytlData && xbrData && ybrData) ? [
                        Number((xtlData as ArrayLike<number>)[0]),
                        Number((ytlData as ArrayLike<number>)[0]),
                        Number((xbrData as ArrayLike<number>)[0]),
                        Number((ybrData as ArrayLike<number>)[0]),
                    ] as [number, number, number, number] : null,
                }));
            }).then(({
                dims, maskData, iouData, lowResData, decoderBounds,
            }) => {
                // dims = [1, C, H, W]
                const channels = dims[1];
                const height = dims[2];
                const width = dims[3];

                // Select the best channel via argmax(iou) when multimask; else channel 0.
                let best = 0;
                if (iouData && channels > 1) {
                    let bestValue = -Infinity;
                    for (let c = 0; c < channels; c++) {
                        const value = Number(iouData[c]);
                        if (value > bestValue) {
                            bestValue = value;
                            best = c;
                        }
                    }
                }

                const planeSize = height * width;
                const maskOffset = best * planeSize;
                const fullMask = new Uint8Array(planeSize);
                let xtl = width;
                let ytl = height;
                let xbr = -1;
                let ybr = -1;
                for (let i = 0; i < planeSize; i++) {
                    if (Number(maskData[maskOffset + i]) > 0) {
                        fullMask[i] = 1;
                        const x = i % width;
                        const y = Math.floor(i / width);
                        if (x < xtl) xtl = x;
                        if (x > xbr) xbr = x;
                        if (y < ytl) ytl = y;
                        if (y > ybr) ybr = y;
                    }
                }

                // CVAT mask RLE encodes the mask cropped to its bounding box; the appended
                // bounds [left, top, right, bottom] are inclusive and the decoder restores a
                // (right-left+1) x (bottom-top+1) mask (see cvat-core rle-utils/annotations-objects).
                let bounds: [number, number, number, number];
                let binary: Uint8Array;
                if (decoderBounds) {
                    // large: `masks` is already cropped to its bbox and the absolute bounds come
                    // from the decoder. Use the mask plane as-is (its size already equals the crop).
                    bounds = decoderBounds;
                    binary = fullMask;
                } else if (xbr < 0) {
                    // base_plus: no foreground -> empty mask.
                    bounds = [0, 0, 0, 0];
                    binary = new Uint8Array(0);
                } else {
                    // base_plus: `masks` covers the full image, so crop it to the derived bbox to
                    // keep the RLE length consistent with the bounds (else CVAT throws
                    // "offset is out of bounds").
                    bounds = [xtl, ytl, xbr, ybr];
                    const cropWidth = xbr - xtl + 1;
                    const cropHeight = ybr - ytl + 1;
                    binary = new Uint8Array(cropWidth * cropHeight);
                    for (let y = 0; y < cropHeight; y++) {
                        const srcRow = (ytl + y) * width + xtl;
                        const dstRow = y * cropWidth;
                        for (let x = 0; x < cropWidth; x++) {
                            binary[dstRow + x] = fullMask[srcRow + x];
                        }
                    }
                }

                // Slice the selected channel of low_res_masks for the next refinement step.
                let lowResMasks: Float32Array | null = null;
                if (lowResData) {
                    const lowResPlane = lowResData.length / channels;
                    lowResMasks = lowResData.slice(best * lowResPlane, (best + 1) * lowResPlane);
                }

                postMessage({
                    action: WorkerAction.DECODE,
                    payload: {
                        masks: binary,
                        lowResMasks,
                        bounds,
                    } as SAM2OutputItem,
                });
            }).catch((error: unknown) => {
                postMessage({ action: WorkerAction.DECODE, error: errorToMessage(error) });
            });
        }
    };
}
