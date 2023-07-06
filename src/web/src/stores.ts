import { writable } from 'svelte/store';

export const mu = writable([] as number[]);
export const lower = writable([] as number[]);
export const upper = writable([] as number[]);
export const covariance_matrix = writable([] as number[][]);