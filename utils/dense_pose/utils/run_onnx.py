import onnx
import torch
import onnxruntime

class Onnx():

    def __init__(self):
        self.outs_names = {"image_shape":None,
                            "pred_boxes":{"tensor":None},
                            "scores":None,
                            "pred_classes":None,
                            "pred_densepose":{"coarse_segm":None,
                            "fine_segm":None,
                            "u":None,
                            "v":None}}
        self.i_counter = 0

    def run_onnx(self,model,input):
        session = model
        inputs = {"image": input.cpu().numpy(),}
        outputs = session.run(None, inputs)
        outs = self.fill_out(outputs,self.outs_names)
        return outs
        # print(outs)

    def fill_out(self,output,outs_names):
        out = {}
        for key, value in outs_names.items():
            if key == "image_shape":
                out[key] = (output[self.i_counter][0],output[self.i_counter][1])
                self.i_counter += 1
            else:
                if isinstance(value,dict):
                    out[key] = self.fill_out(output,outs_names[key])
                else:
                    out[key] = torch.from_numpy(output[self.i_counter])
                    self.i_counter += 1
        return out    




    def print_onnx_model_io(model_path):
        # Load the ONNX model
        model = onnx.load(model_path)
        # Check the model
        onnx.checker.check_model(model)
        print(f"Model loaded from: {model_path}")

        # Get model graph
        graph = model.graph

        print("Inputs:")
        for input_tensor in graph.input:
            name = input_tensor.name
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append('?')
            dtype = input_tensor.type.tensor_type.elem_type
            print(f"  Name: {name}, Shape: {shape}, Type: {dtype}")

        print("\nOutputs:")
        for output_tensor in graph.output:
            name = output_tensor.name
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append('?')
            dtype = output_tensor.type.tensor_type.elem_type
            print(f"  Name: {name}, Shape: {shape}, Type: {dtype}")