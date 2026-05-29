// Copyright (C) 2023-2024 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { LRUCache } from 'lru-cache';
import {
    CVATCore, MLModel, Job, InteractorResults,
    Source, ShapeType,
} from 'cvat-core-wrapper';
import { PluginEntryPoint, APIWrapperEnterOptions, ComponentBuilder } from 'components/plugins-entrypoint';
import {
    InitBody, DecodeBody, WorkerAction, SAM2OutputItem,
} from './inference.worker';

interface SAM2Plugin {
    name: string;
    description: string;
    cvat: {
        lambda: {
            call: {
                enter: (
                    plugin: SAM2Plugin,
                    taskID: number,
                    model: MLModel,
                    args: any,
                ) => Promise<null | APIWrapperEnterOptions>;
                leave: (
                    plugin: SAM2Plugin,
                    result: object,
                    taskID: number,
                    model: MLModel,
                    args: any,
                ) => Promise<any>;
            };
        };
        jobs: {
            get: {
                leave: (
                    plugin: SAM2Plugin,
                    results: any[],
                    query: { jobID?: number }
                ) => Promise<any>;
            };
        };
    };
    data: {
        initialized: boolean;
        worker: Worker;
        core: CVATCore | null;
        jobs: Record<number, Job>;
        modelID: string;
        modelURL: string;
        embeddings: LRUCache<string, Float32Array>;
        features0: LRUCache<string, Float32Array>;
        features1: LRUCache<string, Float32Array>;
        lowResMasks: LRUCache<string, Float32Array>;
        lastClicks: ClickType[];
    };
    callbacks: {
        mask2Rle: ((points: ArrayLike<number>) => number[]) | null;
    };
}

interface ClickType {
    clickType: 0 | 1 | 2 | 3;
    x: number;
    y: number;
}

function getModelScale(w: number, h: number): { width: number; height: number; scaleX: number; scaleY: number } {
    const targetSize = 1024;
    const scaleX = targetSize / w;
    const scaleY = targetSize / h;
    return { scaleX, scaleY, width: w, height: h };
}

function modelData(
    {
        clicks, imageEmbed, highResFeats0, highResFeats1, modelScale, maskInput,
    }: {
        clicks: ClickType[];
        imageEmbed: Float32Array;
        highResFeats0: Float32Array;
        highResFeats1: Float32Array;
        modelScale: { width: number; height: number; scaleX: number; scaleY: number };
        maskInput: Float32Array | null;
    },
): DecodeBody {
    const n = clicks.length;
    const pointCoords = new Float32Array(2 * n);
    const pointLabels = new Float32Array(n);

    // Scale and add clicks
    for (let i = 0; i < clicks.length; i++) {
        pointCoords[2 * i] = clicks[i].x * modelScale.scaleX;
        pointCoords[2 * i + 1] = clicks[i].y * modelScale.scaleY;
        pointLabels[i] = clicks[i].clickType;
    }

    return {
        imageEmbed,
        highResFeats0,
        highResFeats1,
        pointCoords,
        pointLabels,
        width: modelScale.width,
        height: modelScale.height,
        maskInput,
    };
}

const sam2Plugin: SAM2Plugin = {
    name: 'Segment Anything 2.1',
    description: 'Handles non-default SAM2 serverless function output',
    cvat: {
        jobs: {
            get: {
                async leave(
                    plugin: SAM2Plugin,
                    results: any[],
                    query: { jobID?: number },
                ): Promise<any> {
                    if (typeof query.jobID === 'number') {
                        [plugin.data.jobs[query.jobID]] = results;
                    }
                    return results;
                },
            },
        },
        lambda: {
            call: {
                async enter(
                    plugin: SAM2Plugin,
                    taskID: number,
                    model: MLModel, { frame }: { frame: number },
                ): Promise<null | APIWrapperEnterOptions> {
                    return new Promise((resolve, reject) => {
                        function resolvePromise(): void {
                            const key = `${taskID}_${frame}`;
                            const hasAllFeatures = (
                                plugin.data.embeddings.has(key) &&
                                plugin.data.features0.has(key) &&
                                plugin.data.features1.has(key)
                            );
                            if (hasAllFeatures) {
                                resolve({ preventMethodCall: true });
                            } else {
                                resolve(null);
                            }
                        }

                        if (model.id === plugin.data.modelID) {
                            if (!plugin.data.initialized) {
                                sam2Plugin.data.worker.postMessage({
                                    action: WorkerAction.INIT,
                                    payload: {
                                        decoderURL: sam2Plugin.data.modelURL,
                                    } as InitBody,
                                });

                                sam2Plugin.data.worker.onmessage = (e: MessageEvent) => {
                                    if (e.data.action !== WorkerAction.INIT) {
                                        reject(new Error(
                                            `Caught unexpected action response from worker: ${e.data.action}`,
                                        ));
                                    }

                                    if (!e.data.error) {
                                        sam2Plugin.data.initialized = true;
                                        resolvePromise();
                                    } else {
                                        reject(new Error(`SAM 2.1 worker was not initialized. ${e.data.error}`));
                                    }
                                };
                            } else {
                                resolvePromise();
                            }
                        } else {
                            resolve(null);
                        }
                    });
                },

                async leave(
                    plugin: SAM2Plugin,
                    result: any,
                    taskID: number,
                    model: MLModel,
                    {
                        frame, pos_points, neg_points, obj_bbox,
                    }: {
                        frame: number, pos_points: number[][], neg_points: number[][], obj_bbox: number[][],
                    },
                ): Promise<InteractorResults | unknown> {
                    return new Promise((resolve, reject) => {
                        if (model.id !== plugin.data.modelID) {
                            resolve(result);
                            return;
                        }

                        const job = Object.values(plugin.data.jobs).find((_job) => (
                            _job.taskId === taskID && frame >= _job.startFrame && frame <= _job.stopFrame
                        )) as Job;

                        if (!job) {
                            throw new Error('Could not find a job corresponding to the request');
                        }

                        plugin.data.jobs = {
                            // we do not need to store old job instances
                            [job.id]: job,
                        };

                        job.frames.get(frame)
                            .then(({ height: imHeight, width: imWidth }: { height: number; width: number }) => {
                                const key = `${taskID}_${frame}`;

                                if (result) {
                                    const encodedImageEmbed = window.atob(result.image_embed);
                                    const encodedFeat0 = window.atob(result.high_res_feats_0);
                                    const encodedFeat1 = window.atob(result.high_res_feats_1);

                                    const uint8ArrayImageEmbed = new Uint8Array(encodedImageEmbed.length);
                                    const uint8ArrayFeat0 = new Uint8Array(encodedFeat0.length);
                                    const uint8ArrayFeat1 = new Uint8Array(encodedFeat1.length);

                                    for (let i = 0; i < encodedImageEmbed.length; i++) {
                                        uint8ArrayImageEmbed[i] = encodedImageEmbed.charCodeAt(i);
                                    }

                                    for (let i = 0; i < encodedFeat0.length; i++) {
                                        uint8ArrayFeat0[i] = encodedFeat0.charCodeAt(i);
                                    }

                                    for (let i = 0; i < encodedFeat1.length; i++) {
                                        uint8ArrayFeat1[i] = encodedFeat1.charCodeAt(i);
                                    }

                                    const float32ArrImageEmbed = new Float32Array(uint8ArrayImageEmbed.buffer);
                                    const float32ArrFeat0 = new Float32Array(uint8ArrayFeat0.buffer);
                                    const float32ArrFeat1 = new Float32Array(uint8ArrayFeat1.buffer);

                                    plugin.data.embeddings.set(key, float32ArrImageEmbed);
                                    plugin.data.features0.set(key, float32ArrFeat0);
                                    plugin.data.features1.set(key, float32ArrFeat1);
                                }

                                const modelScale = getModelScale(imWidth, imHeight);

                                const clicks: ClickType[] = [];
                                if (obj_bbox.length) {
                                    clicks.push({ clickType: 2, x: obj_bbox[0][0], y: obj_bbox[0][1] });
                                    clicks.push({ clickType: 3, x: obj_bbox[1][0], y: obj_bbox[1][1] });
                                }

                                pos_points.forEach((point) => {
                                    clicks.push({ clickType: 1, x: point[0], y: point[1] });
                                });

                                neg_points.forEach((point) => {
                                    clicks.push({ clickType: 0, x: point[0], y: point[1] });
                                });

                                const isLowResMaskSuitable = JSON
                                    .stringify(clicks.slice(0, -1)) === JSON.stringify(plugin.data.lastClicks);
                                const feeds = modelData({
                                    clicks,
                                    imageEmbed: plugin.data.embeddings.get(key) as Float32Array,
                                    highResFeats0: plugin.data.features0.get(key) as Float32Array,
                                    highResFeats1: plugin.data.features1.get(key) as Float32Array,
                                    modelScale,
                                    maskInput: isLowResMaskSuitable ? plugin.data.lowResMasks.get(key) || null : null,
                                });

                                plugin.data.worker.postMessage({
                                    action: WorkerAction.DECODE,
                                    payload: feeds,
                                });

                                plugin.data.worker.onmessage = ((e) => {
                                    if (e.data.action !== WorkerAction.DECODE) {
                                        const error = 'Caught unexpected action response from worker: ' +
                                                `${e.data.action}, while "${WorkerAction.DECODE}" was expected`;
                                        reject(new Error(error));
                                    }

                                    if (!e.data.error) {
                                        const { masks, lowResMasks, bounds } = e.data.payload as SAM2OutputItem;
                                        plugin.data.lowResMasks.set(key, lowResMasks);
                                        plugin.data.lastClicks = clicks;

                                        let rle = plugin.callbacks.mask2Rle!(masks);
                                        if (rle.length < 2) {
                                            rle = [0, 0, 0, 0, 0];
                                        } else {
                                            rle.push(...bounds);
                                        }

                                        resolve({
                                            shapes: [{
                                                points: Int32Array.from(rle),
                                                group: 0,
                                                source: Source.SEMI_AUTO,
                                                occluded: false,
                                                rotation: 0,
                                                type: ShapeType.MASK,
                                                attributes: [],
                                            }],
                                        } as InteractorResults);
                                    } else {
                                        reject(new Error(`Decoder error. ${e.data.error}`));
                                    }
                                });

                                plugin.data.worker.onerror = ((error) => {
                                    reject(error);
                                });
                            });
                    });
                },
            },
        },
    },
    data: {
        initialized: false,
        core: null,
        worker: new Worker(new URL('./inference.worker', import.meta.url)),
        jobs: {},
        modelID: 'ort-facebookresearch-sam2-hiera-large',
        modelURL: '/assets/sam2.1_hiera_large.decoder.onnx',
        embeddings: new LRUCache({
            // float32 tensor [256, 64, 64] is 4 MB, max 128 MB
            max: 32,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
        features0: new LRUCache({
            // float32 tensor [32, 256, 256] is 8 MB, max 128 MB
            max: 32,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
        features1: new LRUCache({
            // float32 tensor [64, 128, 128] is 4 MB, max 128 MB
            max: 32,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
        lowResMasks: new LRUCache({
            // float32 tensor [1, 256, 256] is 0.25 MB, max 8 MB
            max: 32,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
        lastClicks: [],
    },
    callbacks: {
        mask2Rle: null,
    },
};

const builder: ComponentBuilder = ({ core }) => {
    sam2Plugin.data.core = core;
    sam2Plugin.callbacks.mask2Rle = core.utils.mask2Rle;
    core.plugins.register(sam2Plugin);

    return {
        name: sam2Plugin.name,
        destructor: () => {
            sam2Plugin.data.embeddings.clear();
            sam2Plugin.data.features0.clear();
            sam2Plugin.data.features1.clear();
            sam2Plugin.data.lowResMasks.clear();
            sam2Plugin.data.worker.terminate();
            sam2Plugin.data.lastClicks = [];
            sam2Plugin.data.jobs = {};
            sam2Plugin.data.core = null;
            sam2Plugin.data.initialized = false;
        },
    };
};

function register(): void {
    if (Object.prototype.hasOwnProperty.call(window, 'cvatUI')) {
        (window as any as { cvatUI: { registerComponent: PluginEntryPoint } })
            .cvatUI.registerComponent(builder);
    }
}

window.addEventListener('plugins.ready', register, { once: true });
