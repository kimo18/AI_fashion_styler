from utils.dense_pose.utils.run_onnx import Onnx
from utils.dense_pose.utils.save_dense_image import Saver
from utils.dense_pose.utils.postprocessor import PostProcess ,PreProcessor
from utils.dense_pose.utils.reader import ImageReader




NewH , New_W = (1200, 800)


class DensePoseGenerator():
    def __init__(self,input_path):
        self.input_path = input_path
        self.original_image = ImageReader.read_image(self.input_path, format="BGR")

    def generate(self, onnx_model):
        results = []
        for image in self.original_image:
            inputs = PreProcessor.pre_process(image, NewH, New_W)
            out = Onnx().run_onnx(onnx_model, inputs[0]["image"])
            result = PostProcess._postprocess([out],inputs,[(NewH , New_W)])
            results.append(result)
        return results

    
    def save(self, results,out_path,image_name):
        for image , result in zip(self.original_image,results):
            entry = {"file_name":self.input_path,"image":image}
            Saver.save_denseimage(entry,result[0]["instances"],out_path,image_name)
        



