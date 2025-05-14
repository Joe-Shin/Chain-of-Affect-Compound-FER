# ------Code that captures live video from webcam, detects face, takes the cropped face image to determine compound emotion, then displays bounding box and emotion on frame------
# Press 'q' to end program.

import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import torch
import os
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoProcessor, AutoConfig
import warnings
from peft import PeftModel
from PIL import Image
import threading
import queue
import cv2


OUTPUT_FLAG=0 # 0 if we don't want to record output video, 1 if we do

# -------------------------- Define models part --------------------------

model_path = 'lora_vision_all_ft_only'
model_base = 'microsoft/Phi-3.5-vision-instruct'

model_paths = model_path.split("/")
model_name = model_paths[-1]

load_8bit=False
load_4bit=False
device_map="auto"
device="cuda"

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# load finetuned model
# This code is borrowed from LLaVA
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map}
    
    if device != "cuda":
        kwargs['device_map'] = {"":device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    #if use_flash_attn:
    #    kwargs['_attn_implementation'] = 'flash_attention_2'

    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config
        processor = AutoProcessor.from_pretrained(model_base, trust_remote_code=True)
        print('Loading Phi3-Vision from base model...')
        model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, trust_remote_code=True, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Phi3-Vision weights...')
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_state_dict.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)

        print('Merging LoRA weights...')
        
        model = model.merge_and_unload()
        
        print('Model Loaded!!!')
    
    else:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)

    return processor, model

# -------------------------- Define dicts, inference function, and other variables --------------------------

# all basic emotions list
edic_basic = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

# two basic emotions -> compound emotion dict
edic_cmpd_rev = {('Happiness', 'Surprise'):'Happily Surprised',
                 ('Happiness', 'Disgust'):'Happily Disgusted',
                 ('Sadness', 'Fear'):'Sadly Fearful',
                 ('Sadness', 'Anger'):'Sadly Angry',
                 ('Sadness', 'Surprise'):'Sadly Surprised',
                 ('Sadness', 'Disgust'):'Sadly Disgusted',
                 ('Fear','Anger'):'Fearfully Angry',
                 ('Fear','Surprise'):'Fearfully Surprised',
                 ('Anger','Surprise'):'Angrily Surprised',
                 ('Anger','Disgust'):'Angrily Disgusted',
                 ('Disgust', 'Surprise'):'Disgustedly Surprised'}

# compound FER inference for single image function
def do_cmpd(image, processor, model):
    
    # image = Image.open(imagepath).convert("RGB")

    inp = f"<|image_1|>\nSelect an emotion label from the following options: 'Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger' or 'Neutral'."
    messages =[ {"role": "user", "content": inp} ]

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to(device)

    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs, 
            max_new_tokens= 1000,
            temperature= 0,
            repetition_penalty= 1.0,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    lb1 = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    

    lb_cand = [e for e in edic_basic if e != lb1]
    lb_cand = ', '.join( lb_cand[:-1] ) + ', or ' + lb_cand[-1]
    
    inp = f"<|image_1|>\nThe primary emotion conveyed by this image was {lb1}. Please select the next most strongly felt emotion from the following options: {lb_cand}."
    # inp = f"<|image_1|>\nThe primary emotion conveyed by this image was {lb1}. Please select the next most strongly felt emotion from the following options: {lb_cand}. If the secondary emotion is not highly apparent, say 'Neutral'."
    messages =[ {"role": "user", "content": inp} ]
    
    # print(inp)

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to(device)

    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs, 
            max_new_tokens= 1000,
            temperature= 0,
            repetition_penalty= 1.0,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    lb2 = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    
    return lb1, lb2

# -------------------------- Multithreading definition --------------------------

# stop flag, for when teminating the code
stop_flag = threading.Event()

frame_queue = queue.Queue(maxsize=1)  # To store frames for inference
result_queue = queue.Queue(maxsize=1)  # To store the latest prediction result
bbox_queue = queue.Queue(maxsize=1)  # To store bounding box coordinates

def display_frames(model_y):
    prediction = ''
    cap = cv2.VideoCapture(0)  # Access the webcam
    x1, y1, x2, y2 = 0, 0, 0, 0

    # -------------------------- Output Video Setting Part, in case we want to record the output video --------------------------
    # Get the video width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    if OUTPUT_FLAG==1:
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the latest prediction result (if available)
        if not result_queue.empty():
            prediction = result_queue.get_nowait()
            # Overlay the prediction on the frame
        cv2.putText(frame, f"{prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # get bounding boxes
        results_y=model_y(frame,verbose=False)

        for result_y in results_y:
            # organize the box coordinates
            boxes=result_y.boxes
            if len(boxes)==0:
                continue
            x1,y1,x2,y2=boxes.xyxy[0][0:4]
            x1=int(x1.item())
            y1=int(y1.item())
            x2=int(x2.item())
            y2=int(y2.item())

        # save the bounding box coordinates
        if bbox_queue.full():
            bbox_queue.get() # Remove the old bounding box
        bbox_queue.put([x1,y1,x2,y2])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Write the frame to the video file
        if OUTPUT_FLAG==1:
            out.write(frame)

        # Display the frame
        cv2.imshow("Webcam", frame)
        
        # Send the frame to the inference thread (if there's space in the queue)
        if not frame_queue.full():
            frame_queue.put(frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()
            break
    
    cap.release()
    cv2.destroyAllWindows()

def run_inference(processor, model):
    while not stop_flag.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Get the most recent bounding box from the queue
            if not bbox_queue.empty():
                bboxes = bbox_queue.get_nowait()  # Retrieve without blocking
                # Draw the bounding box (replace with your logic)
                x1, y1, x2, y2 = bboxes

            # crop the face image to put in the model
            face_img=frame[y1:y2,x1:x2,:]

            # convert frame from cv2 bgr format to PIL rgb format
            image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            # Perform inference

            lb1, lb2 = do_cmpd(image, processor, model)
            pred_comp_emo = ''
            for ind,tog in edic_cmpd_rev.items():
                if lb1 in ind and lb2 in ind:
                    pred_comp_emo = tog
            if lb1 == 'Neutral':
                pred_comp_emo = 'Neutral'
            elif lb2 == 'Neutral':
                pred_comp_emo = lb1
            elif pred_comp_emo not in edic_cmpd_rev.values():
                pred_comp_emo = f"OTHER PREDICTION: {lb1}, {lb2}"
                
            print(f"{pred_comp_emo}")

            if result_queue.full():
                result_queue.get() # Remove old result
            result_queue.put(pred_comp_emo)

# -------------------------- Main code --------------------------

if __name__ == "__main__":

    # yolo model load
    model_y=YOLO('yolov8n-face.pt')

    # load LORA finetuned processor and model (phi3.5 vision)
    disable_torch_init()
    processor, model = load_pretrained_model(model_path = model_path, model_base=model_base, 
                                            model_name=model_name, device_map=device, 
                                            load_4bit=load_4bit, load_8bit=load_8bit,
                                            device=device, use_flash_attn=True
    )

    # Start threads
    display_thread = threading.Thread(target=display_frames, args = (model_y,))
    inference_thread = threading.Thread(target=run_inference, args=(processor, model))
    
    display_thread.start()
    inference_thread.start()
    
    display_thread.join()
    inference_thread.join()

    print("Program terminated.")
