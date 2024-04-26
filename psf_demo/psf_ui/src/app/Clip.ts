export interface Clip {
    id: number;
    path: string;
    duration?: number;
    dataset: string;
    best_path?: string;
    best_cos?: number;
    best_asd_label?: string;
    second_path?: string;
    second_cos?: number;
    second_asd_label?: string;
    hundred_path?: string;
    hundred_cos?: number;
    hundred_asd_label?: string;
    five_hundred_path?: string;
    five_hundred_cos?: number;
    five_hundred_asd_label?: string;
}