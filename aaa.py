import torch
from languagebind import LanguageBindAudio, LanguageBindAudioTokenizer, LanguageBindAudioProcessor, AutoConfig

pretrained_ckpt = 'LanguageBind/LanguageBind_Audio_FT'  # also 'LanguageBind/LanguageBind_Audio'
configs = LanguageBindAudio.from_pretrained(pretrained_ckpt)
print(configs)