import sys
sys.path.append('/home/slionar/00_eth/internship/venomlizard/VLT5/src')
sys.path.append('/home/slionar/00_eth/internship/venomlizard/VLT5/inference')

from param import parse_args

args = parse_args(
    parse=False,
    backbone='t5-base',
    load='VLT5/overfit2/Epoch30'
)
args.gpu = 0

from vqa import Trainer

trainer = Trainer(args, train=False)

# FRCNN
from VLT5.inference.processing_image import Preprocess
from VLT5.inference.visualizing_image import SingleImageViz
from VLT5.inference.modeling_frcnn import GeneralizedRCNN
from VLT5.inference.utils import Config, get_data

#image_filename = ["input.jpg"]
image_filename = ["fireplace2.jpg"]
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)

image_dirname = image_filename
images, sizes, scales_yx = image_preprocess(image_filename)

output_dict = frcnn(
    images,
    sizes,
    scales_yx = scales_yx,
    padding = 'max_detections',
    max_detections = frcnn_cfg.max_detections,
    return_tensors = 'pt'
)

normalized_boxes = output_dict.get("normalized_boxes")
features = output_dict.get("roi_features")

from tokenization import VLT5TokenizerFast

tokenizer = VLT5TokenizerFast.from_pretrained('t5-base')

questions = ["vqa: What is the main doing?",
             "vqa: What color is the clothing the man wears?",
             "vqa: What color is the horse?",]

questions = ["fire fire fire"]

input_ids = tokenizer(questions[0], return_tensors='pt', padding=True).input_ids
batch = {}
batch['input_ids'] = input_ids
batch['vis_feats'] = features
batch['boxes'] = normalized_boxes

result = trainer.model.test_step(batch)
print(f"Q: {questions[0]}")
print(f"A: {result['pred_ans'][0]}")