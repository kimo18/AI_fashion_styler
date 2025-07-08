import os
import onnxruntime as ort
from utils.humanparse_preprocess import ImageReader
from utils.landmark_json import Landmark_Json
from utils.cloth_mask import ClothMask
from utils.get_parse_agnostic import Parse_Agnostic
from Hr_Viton_inferene.test_generator import GenerateStyle
from utils.extract_meta_data import GROQ_METADATA
from utils.llama2 import LLAMA_Styler
from PIL import Image
import cv2

 
data_path = os.path.join(".","Data")
    
image_dir = os.path.join(".","Data","image")
clothes_dir = os.path.join(".","Data","cloth")

human_parsing_dir = os.path.join(".","Data","image-parse-v3")
human_parsing_agnosti_dir = os.path.join(".","Data","image-parse-agnostic-v3.2")

clothes_dir = os.path.join(".","Data","cloth")
clothes_mask_dir = os.path.join(".","Data","cloth-mask")

openpose_json_dir = os.path.join(".","Data","openpose_json")

togc_dir = os.path.join(".","onnx","hr-viton","tocg.onnx") #HRVITON related
gen_dir = os.path.join(".","onnx","hr-viton","gen.onnx") #HRVITON related


output_clothes_json = os.path.join(".","Data","meta_data","clothes.json")
output_persons_json = os.path.join(".","Data","meta_data","persons.json")

onnx_dir = "onnx"

generate = False
if generate:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(os.path.join(".",onnx_dir,"humanparse","PGN_nodropout_tester.onnx"),sess_options=options, providers=['CPUExecutionProvider'])
    session_dense = ort.InferenceSession(os.path.join(".",onnx_dir,"denseparse","densepose_full.onnx"),sess_options=options, providers=['CPUExecutionProvider'])
    session_cloth_mask  = ort.InferenceSession(os.path.join(".",onnx_dir,"cloth_mask","u2net.onnx"),sess_options=options, providers=['CPUExecutionProvider'])



    ################HUMANPARSE##############################
    # read image from Image_dir
    imagereader = ImageReader(image_dir)
    image , image_name = imagereader.read_batch()
    #run humanpase inference
    input_name = session.get_inputs()[0].name
    input_dict = {
        input_name: image[0]                #1,3,612,408
    }
    outputs = session.run(None, input_dict)
    parsing_im =imagereader.decode_labels(outputs[0],num_classes = 20)
    array_2d = outputs[0].squeeze()
    #save
    cv2.imwrite('{}/{}.png'.format(human_parsing_dir, image_name), array_2d)
    parsing_im.save('{}/{}_vis.png'.format(human_parsing_dir, image_name))

    ############################LANDMARKJSON################
    landmark_json = Landmark_Json(image_dir,image_name,openpose_json_dir)
    landmark_json.get_landmarks_json()

    #########################DENSEPARSE#######################

    #########################ClOTHMASK########################
    cloth_mask = ClothMask(clothes_dir)
    input_process , cloth_name = cloth_mask.process_image()
    ort_inputs = {"input": input_process}
    ort_outs = session_cloth_mask.run(None, ort_inputs)
    binary = cloth_mask.postprocess(ort_outs, cloth_mask.orig_size)
    cleaned = cloth_mask.clean_mask(binary)
    #Save  
    Image.fromarray(cleaned).save('{}/{}.jpg'.format(clothes_mask_dir, cloth_name))
    ########################PARSEAGNOSTIC#####################
    agnostic = Parse_Agnostic(data_path)
    agtnostic_output, parse_name = agnostic.run_agnostic()

    #Save         
    agtnostic_output.save(os.path.join(human_parsing_agnosti_dir, parse_name))
    

####################METADATA########################
if False:
    metadata_generator = GROQ_METADATA(clothes_dir,image_dir,output_clothes_json,output_persons_json)
    metadata_generator.generate_meta_data()

##################################LLAMAV2###############
styler = LLAMA_Styler(output_persons_json,output_clothes_json)
styler.style()


########################HRVITON##########################
generate_style = GenerateStyle(togc_dir,gen_dir)
generate_style.Generate()








