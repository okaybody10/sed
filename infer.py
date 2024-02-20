import torch
import os
import subprocess
import torchaudio
import librosa
import argparse
import json
import re
import copy
import pandas as pd
import sed_scores_eval
from sed_scores_eval import collar_based, intersection_based
from languagebind import LanguageBindAudio, LanguageBindAudioTokenizer, LanguageBindAudioProcessor
from collections import OrderedDict

def make_datas(filter_label, audio_duration) :
    if len(filter_label) == 0 :
        return None, None
    convert_label = {key: [i[key] for i in filter_label] for key in filter_label[0]}
    filter_conv = {key : convert_label[key] for key in convert_label if key not in ['구분', '이유', '이유 상세', '주요 태그', '제거 키워드', '세그먼트']}
    print(filter_conv)
    df = pd.DataFrame(filter_conv)
    df = df[['filename', 'onset', 'offset', 'event_label']]
    df2 = pd.DataFrame({'filename' : [filter_conv['filename'][0]], 'duration' : [audio_duration]})
    return df, df2

def gt_label(file_path, audio_duration) :
    label_list = []
    label_path = os.path.splitext(file_path)[0] + ".txt"
    with open(label_path, "r") as fp :
        cnt = 0
        label_dict = {}
        for cnt, line in enumerate(fp) :
            if cnt % 8 == 7 :
                label_list.append(label_dict)
                label_dict = {}
                continue
            elif cnt % 8 == 0 :
                l = re.findall(r'[0-9]{2}', line.strip())
                onset = int(l[0]) * 3600 + int(l[1]) * 60 + int(l[2])
                offset = int(l[4]) * 3600 + int(l[5]) * 60 + int(l[6])
                label_dict['onset'] = onset
                label_dict['offset'] = offset
            else :
                vv = line.split(':')
                label_dict[vv[0]] = vv[1].strip()
    fp.close()
    filter_label = []
    for label in label_list :
        filter_str = label['이유 상세']
        # Ground_truth condition
        if 'bgm' in filter_str or '효과음' in filter_str :
            label['event_label'] = 'music'
            label['filename'] = os.path.basename(file_name)
            filter_label.append(label)
    ground_truth, metadata = make_datas(filter_label, audio_duration)
    ground_truth.to_csv(os.path.splitext(file_path)[0] + "_ground.tsv", sep='\t')
    metadata.to_csv(os.path.splitext(file_path)[0] + "_meta.tsv", sep='\t')
    # return make_datas(filter_label, audio_duration)

def batch(file_path:str, batch_duration:int = 2):
    """
        Split audio with duration
        file_path: path to audio
        batch_duration: duration to split, default is 2 seconds
    """
    split_result = []
    audio, sample_rate = torchaudio.load(file_path)
    audio_duration = librosa.get_duration(path=file_path, sr = sample_rate)
    # audio, sample_rate = torchaudio.load(file_path)

    # print(audio_duration)

    start_duration = 0
    while True:
        end_duration = start_duration + batch_duration
    
        is_stop = end_duration >= audio_duration
    
        if not is_stop:  
            split_result.append(audio[:, start_duration * sample_rate: end_duration * sample_rate])
            start_duration = end_duration
        else:
            if len(audio[start_duration * sample_rate :]) < 400 :
                pass
            else :
                split_result.append(audio[:, start_duration * sample_rate:]) 
            break
    return split_result, audio_duration

def collect_result(raw) :
    before = raw[0]
    result_tmp = []
    start = 0
    for i, raw_result in enumerate(raw) :
        if set(raw_result) == set(before) :
            pass
        else :
            result_tmp.append(
                {
                    "timestamps": [start, i], "tags" : before
                }
            )
            start = i
            before = raw_result
    result_tmp.append({
        "timestamps" : [start, len(raw)], "tags" : before
    })
    return result_tmp



if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--model", default="LanguageBind/LanguageBind_Audio", help= "Name of using model")
    parser.add_argument("--checkpoint_path", help= "Path of checkpoint file")
    parser.add_argument("--file", required=True, help= "Input file")
    parser.add_argument("--split_time", default=1, type=int, help="Split time duration when audio preprocessing")
    parser.add_argument("--k", default=2, type=int, )
    parser.add_argument("--output", help="output path")
    parser.add_argument("--test", action='store_true', help="If measure metric using test set, set True")
    args = parser.parse_args()

    pretrained_ckpt = args.model  # also 'LanguageBind/LanguageBind_Audio_FT'
    checkpoint_path = args.checkpoint_path if args.checkpoint_path is not None else ""
    checkpoint_path_name = os.path.basename(args.checkpoint_path).split('.')[0] if args.checkpoint_path is not None else ""
    file_name = os.path.split(args.file)[1].split('.')[0]
    output_file = f"/{checkpoint_path_name}_{file_name}_topk_{args.k}_result.json"
    if args.output is None  :
        args.output = os.path.split(args.file)[0]
    model = LanguageBindAudio.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    tokenizer = LanguageBindAudioTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    audio_process = LanguageBindAudioProcessor(model.config, tokenizer)

    if args.checkpoint_path is not None :
        state_dict = torch.load(checkpoint_path)['state_dict']
        new_state_dict = OrderedDict()
        for n, v in state_dict.items():
            name = n.replace("module.","") # .module이 중간에 포함된 형태라면 (".module","")로 치환
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    if os.path.splitext(args.file)[1] != ".wav" :
        command = "ffmpeg -y -i {} -acodec pcm_s16le -ac 1 -ar 16000 {}.wav".format(args.file, os.path.splitext(args.file)[0])
        args.file = os.path.splitext(args.file)[0] + '.wav'
        subprocess.call(command, shell=True)

    # Inference Part
    print("Enter here!")
    model.eval()
    prompt = ['exclamation', 'sound effect', 'music', 'speech', 'background noise'] 
    # data = audio_process(r"/gallery_tate/jaehyuk.sung/sed/out.wav", prompt, return_tensors='pt')
    process, audio_duration = batch(args.file, args.split_time)
    print("Split end here!")
    data = audio_process(process, prompt, return_tensors='pt') # Time
    print("Data processing end here!")
    # Divides tensors..
    datas = []
    lens = data['pixel_values'].shape[0]

    for i in range(0, lens, 1000) :
        add = copy.deepcopy(data)
        add['pixel_values'] = add['pixel_values'][i : min(i+1000, lens)]
        datas.append(add)
    score_matrix = torch.empty((0, len(prompt)), dtype=torch.float32)
    for i, data in enumerate(datas) :
        print(i)
        with torch.no_grad() :
            results = model(**data)
        score_matrix_part = results.image_embeds @ results.text_embeds.T
        score_matrix = torch.cat((score_matrix, score_matrix_part), 0)
    score_matrix = torch.softmax(score_matrix, dim=-1)
    print(score_matrix)
    topk_results = torch.topk(score_matrix, args.k, dim=1, sorted=True).indices
    # print(torch.take(prompt, topk_results))
    result = [[prompt[i] for i in topk_result] for topk_result in topk_results]
    # print(result, collect_result(result))
    with open(args.output + output_file, "w") as json_file :
        json.dump(collect_result(result), json_file)
    # Validate
    if args.test == True :
        log_result = {}
        timestamps = [i for i in range(score_matrix.shape[0] + 1)]
        gt_label(args.file, audio_duration)
        scores = {}
        scores[file_name] = sed_scores_eval.utils.scores.create_score_dataframe(
            score_matrix.numpy(), timestamps, prompt
        )
        collar = .2
        offset_collar_rate = .2
        time_decimals = 30
        f_best, p_best, r_best, thresholds_best, stats_best = collar_based.best_fscore(
            scores = scores,
            ground_truth = os.path.splitext(args.file)[0] + "_ground.tsv",
            onset_collar=collar, offset_collar=collar,
            offset_collar_rate=offset_collar_rate,
            time_decimals=time_decimals,
            num_jobs=8,
        )
        for cls in f_best:
            print(cls)
            print(' ', 'f:', f_best[cls])
            print(' ', 'p:', p_best[cls])
            print(' ', 'r:', r_best[cls])
            log_result[f'F1 of {cls}'] = f_best[cls]
            log_result[f'Precision of {cls}'] = p_best[cls]
            log_result[f'Recall of {cls}'] = r_best[cls]
        
        dtc_threshold = .7
        gtc_threshold = .7
        cttc_threshold = None
        alpha_ct = .0
        alpha_st = 1.

        psds, (single_class_psds, psd_roc), single_class_psd_rocs = intersection_based.psds(
            scores = scores,
            ground_truth = os.path.splitext(args.file)[0] + "_ground.tsv",
            audio_durations= os.path.splitext(args.file)[0] + "_meta.tsv",
            dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold,
            alpha_ct=alpha_ct, alpha_st=alpha_st,
            unit_of_time='hour', max_efpr=100.,
        )
        log_result['psds'] = psds
        print('PSDS:', psds)
        log_output_file = f"/{checkpoint_path_name}_{file_name}_topk_{args.k}_log_result.json"
        with open(args.output + log_output_file, "w") as jf :
            json.dump(log_result, jf)