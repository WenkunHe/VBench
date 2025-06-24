import os
import json
import numpy as np

import torch
import clip
from tqdm import tqdm
from vbench.utils import load_video, load_dimension_info, clip_transform, read_frames_decord_by_fps, CACHE_DIR
from vbench.third_party.ViCLIP.viclip import ViCLIP
from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)


def get_text_features(model, input_text, tokenizer):
    if isinstance(input_text, list):
        input_text = input_text[0]
    text_template= f"{input_text}"
    with torch.no_grad():
        text_features = model.encode_text(text_template).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)      
    return text_features

def get_vid_features(model, input_frames):
    with torch.no_grad():
        clip_feat = model.encode_vision(input_frames,test=True).float()
        clip_feat /= clip_feat.norm(dim=-1, keepdim=True)    
    return clip_feat

def get_predict_label(clip_feature, text_feats_tensor, top=5):
    label_probs = (100.0 * clip_feature @ text_feats_tensor.T).softmax(dim=-1)
    top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
    return top_probs, top_labels

def temporal_style(clip_model, video_dict, tokenizer, device, sample="middle"):
    sim = []
    video_results = []
    image_transform = clip_transform(224)
    for info in tqdm(video_dict, disable=get_rank() > 0):
        query = info['prompt']
        # text = clip.tokenize([query]).to(device)
        video_list = info['video_list']
        for video_path in video_list:
            cur_video = []
            with torch.no_grad():
                # images = load_video(video_path, num_frames=8)
                images = read_frames_decord_by_fps(video_path, num_frames=8, sample=sample)
                images = image_transform(images)
                images = images.to(device)
                clip_feat = get_vid_features(clip_model,images.unsqueeze(0))
                text_feat = get_text_features(clip_model, query, tokenizer)
                logit_per_text =  clip_feat @ text_feat.T
                score_per_video =  float(logit_per_text[0][0].cpu())
                sim.append(score_per_video)
                video_results.append({'video_path': video_path, 'video_results': score_per_video})
    avg_score = np.mean(sim)
    return avg_score, video_results

def compute_temporal_style(json_list, device, submodules_list, **kwargs):
    tokenizer = SimpleTokenizer(os.path.join(CACHE_DIR, "ViCLIP/bpe_simple_vocab_16e6.txt.gz"))
    viclip = ViCLIP(tokenizer= tokenizer, **submodules_list).to(device)
    _, video_dict = load_dimension_info(json_list, dimension='temporal_style', lang='en')
    video_dict = distribute_list_to_rank(video_dict)
    all_results, video_results = temporal_style(viclip, video_dict, tokenizer, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results


from .utils import ComputeSingleMetric

class ComputeSingleTemporalStyle(ComputeSingleMetric):
    def __init__(self, device, submodules_list):
        super().__init__(device, submodules_list)
        self.tokenizer = SimpleTokenizer(os.path.join(CACHE_DIR, "ViCLIP/bpe_simple_vocab_16e6.txt.gz"))
        self.viclip = ViCLIP(tokenizer=self.tokenizer, **submodules_list).to(device)
        self.image_transform = clip_transform(224)
    
    def update_single(self, images_tensor, prompts):
        clip_model = self.viclip
        tokenizer = self.tokenizer
        image_transform = self.image_transform
        device = self.device

        query = prompts

        with torch.no_grad():
            indices = self.get_frame_indices(num_frames=8, vlen=images_tensor.shape[0])
            images = images_tensor[indices]
            images = image_transform(images)
            images = images.to(device)
            clip_feat = get_vid_features(clip_model, images.unsqueeze(0))
            text_feat = get_text_features(clip_model, query, tokenizer)
            logit_per_text = clip_feat @ text_feat.T
            score_per_video = float(logit_per_text[0][0].cpu())

        self.score += score_per_video
        self.n_samples += 1

    def get_frame_indices(self, num_frames, vlen):
        acc_samples = min(num_frames, vlen)

        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        frame_indices = [(x[0] + x[1]) // 2 for x in ranges]

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices

        return frame_indices
