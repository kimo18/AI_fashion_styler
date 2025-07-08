import torch
import torch.nn.functional as F
import argparse
import os
import time
from Hr_Viton_inferene.cp_dataset_test import CPDatasetTest, CPDataLoader
import kornia  as tgm
from collections import OrderedDict
import onnx
import onnxruntime as ort
import numpy as np
import torch
from PIL import Image



def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        try:
            array = tensor.numpy().astype('uint8')
        except:
            array = tensor.detach().numpy().astype('uint8')

        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        im = Image.fromarray(array)
        im.save(os.path.join(save_dir, img_name), format='JPEG')
        


def make_grid(N, iH, iW):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1)
    
    grid = torch.cat([grid_x, grid_y], 3)
    return grid


def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm
def get_opt():
    # parser = argparse.ArgumentParser()


    # Cuda availability


    opt = {
    'test_name': '',
    'dataroot': "./Data2",
    'datamode': 'test',
    'data_list': "test_pairs.txt",
    'output_dir': "Output",
    'datasetting': "unpaired",
    'fine_width': 768,
    'fine_height': 1024,
    'semantic_nc': 13,
    'output_nc': 13,
    'gen_semantic_nc': 7,  }

    return opt

def load_checkpoint_G(model, checkpoint_path,opt):
    if not os.path.exists(checkpoint_path):
        print("Invalid path!")
        return
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict.items()])
    new_state_dict._metadata = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict._metadata.items()])
    model.load_state_dict(new_state_dict, strict=True)
    



def test(opt, test_loader, tocg, generator):
    gauss = tgm.filters.GaussianBlur2d((15, 15), (3, 3))
    
    # Model
    # tocg.eval()
    # generator.eval()
    
    if opt['output_dir'] is not None:
        out = opt['output_dir']
        output_dir = os.path.join('.',f'{out}')
    else:
        output_dir = os.path.join('./output', opt['test_name'],
                            opt['datamode'], opt['datasetting'], 'generator', 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    num = 0
    iter_start_time = time.time()
    with torch.no_grad():
        for inputs in test_loader.data_loader:

           
            # pose_map = inputs['pose']
            pre_clothes_mask = inputs['cloth_mask'][opt['datasetting']]
            label = inputs['parse']
            parse_agnostic = inputs['parse_agnostic']
            agnostic = inputs['agnostic']
            clothes = inputs['cloth'][opt['datasetting']] # target cloth
            densepose = inputs['densepose']
            im = inputs['image']
            input_label, input_parse_agnostic = label, parse_agnostic
            pre_clothes_mask = torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(np.float32))



            # down
            # pose_map_down = F.interpolate(pose_map, size=(256, 192), mode='bilinear')
            pre_clothes_mask_down = F.interpolate(pre_clothes_mask, size=(256, 192), mode='nearest')
            # input_label_down = F.interpolate(input_label, size=(256, 192), mode='bilinear')
            input_parse_agnostic_down = F.interpolate(input_parse_agnostic, size=(256, 192), mode='nearest')
            # agnostic_down = F.interpolate(agnostic, size=(256, 192), mode='nearest')
            clothes_down = F.interpolate(clothes, size=(256, 192), mode='bilinear')
            densepose_down = F.interpolate(densepose, size=(256, 192), mode='bilinear')

            shape = pre_clothes_mask.shape
            
            # multi-task inputs
            input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
            input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

            # forward
            

            _,_,_,_,flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg.run(None, {'input1': input1.cpu().numpy(),'input2': input2.cpu().numpy()})
            # flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(input1,input2)
            # print(flow_list[-1].shape, fake_segmap.shape,warped_clothmask_paired.shape)

            flow_list = torch.from_numpy(flow_list).to('cpu')
            fake_segmap = torch.from_numpy(fake_segmap).to('cpu')
            # warped_cloth_paired = torch.from_numpy(warped_cloth_paired).to('cpu')
            warped_clothmask_paired = torch.from_numpy(warped_clothmask_paired).to('cpu')
            # warped cloth mask one hot
            
            warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float32))

            
            cloth_mask = torch.ones_like(fake_segmap)
            cloth_mask[:,3:4, :, :] = warped_clothmask_paired
            fake_segmap = fake_segmap * cloth_mask
                    
            # make generator input parse map
            fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(opt['fine_height'], opt['fine_width']), mode='bilinear'))
            fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

            
            old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt['fine_height'], opt['fine_width']).zero_()
            old_parse.scatter_(1, fake_parse, 1.0)

            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            
            parse = torch.FloatTensor(fake_parse.size(0), 7, opt['fine_height'], opt['fine_width']).zero_()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    parse[:, i] += old_parse[:, label]
                    
            # warped cloth
            N, _, iH, iW = clothes.shape
            flow = F.interpolate(flow_list.permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
            flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
            
            grid = make_grid(N, iH, iW)
            warped_grid = grid + flow_norm
            warped_cloth = F.grid_sample(clothes, warped_grid, padding_mode='border')
            warped_clothmask = F.grid_sample(pre_clothes_mask, warped_grid, padding_mode='border')
            # if opt[occlusion:
            #     warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
            #     warped_cloth = warped_cloth * warped_clothmask + torch.ones_like(warped_cloth) * (1-warped_clothmask)
            

            # output = generator(torch.cat((agnostic, densepose, warped_cloth), dim=1), parse)
            gen_input_1 = torch.cat((agnostic, densepose, warped_cloth), dim=1)
            output = generator.run(None, {'input1': gen_input_1.cpu().numpy(),'input2': parse.cpu().numpy()})
            output = torch.from_numpy(output[0]).to('cpu')
            # visualize
            unpaired_names = []
            for i in range(shape[0]):
                unpaired_name = (inputs['c_name']['paired'][i].split('.')[0] + '_' + inputs['c_name'][opt['datasetting']][i].split('.')[0] + '.png')
                unpaired_names.append(unpaired_name)

            save_images(output, unpaired_names, output_dir)
                
            num += shape[0]
            print(num)

    print(f"Test time {time.time() - iter_start_time}")


def generate(session,session_2):
    opt = get_opt()
    print(opt)
    print("Start to test %s!")
    
    # create test dataset & loader
    test_dataset = CPDatasetTest(opt)
    test_loader = CPDataLoader(opt, test_dataset)
    
    # visualization
    # if not os.path.exists(opt[tensorboard_dir):
    #     os.makedirs(opt[tensorboard_dir)
    # board = SummaryWriter(log_dir=os.path.join(opt[tensorboard_dir, opt[test_name, opt[datamode, opt[datasetting))

    ## Model
    # tocg
    # input1_nc = 4  # cloth + cloth-mask
    # input2_nc = opt[semantic_nc + 3  # parse_agnostic + densepose
    # # tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt[output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
       
    # # generator
    # opt[semantic_nc = 7
    # generator = SPADEGenerator(opt, 3+3+3)
    # generator.print_network()
       
    # Load Checkpoint
    # load_checkpoint(tocg, opt[tocg_checkpoint,opt)
    # load_checkpoint_G(generator, opt[gen_checkpoint,opt)

    # tocg_wrapper = TOCGWrapper(tocg)
    # dummy_input1 = torch.randn(1, 4, 256, 192)
    # dummy_input2 = torch.randn(1, 16, 256, 192)

    # # Export to ONNX
    # torch.onnx.export(
    #     tocg_wrapper,
    #     (dummy_input1, dummy_input2),
    #     "tocg.onnx",
    #     input_names=["input1", "input2"],
    #     output_names=["flow", "fake_segmap", "warped_cloth", "warped_clothmask"],
    #     opset_version=16,
    #     dynamic_axes={
    #         "input1": {0: "batch_size"},
    #         "input2": {0: "batch_size"},
    #         "flow": {0: "batch_size"},
    #         "fake_segmap": {0: "batch_size"},
    #         "warped_cloth": {0: "batch_size"},
    #         "warped_clothmask": {0: "batch_size"},
    #     }
    # )
    # generator.eval()
    # generator_wrapper = SPADWrapper(generator)
    # generator_wrapper.eval()
    # dummy_input1 = torch.randn(1, 9, 1024, 768,dtype=torch.float32)
    # dummy_input2 = torch.randn(1, 7, 1024, 768,dtype=torch.float32)

    # # Export to ONNX
    # torch.onnx.export(
    #     generator_wrapper,
    #     (dummy_input1, dummy_input2),
    #     "gen.onnx",
    #     input_names=["input1", "input2"],
    #     output_names=[],
    #     opset_version=13,
    #     do_constant_folding=True,
    #     dynamic_axes= None
    # )
    
    # quit()
    # Train
    test(opt, test_loader, session, session_2)

    print("Finished testing!")




class GenerateStyle():
    def __init__(self,togc_dir, gen_dir):
        self.togc_session = ort.InferenceSession(togc_dir, providers=['CPUExecutionProvider'])
        self.gen_session = ort.InferenceSession(gen_dir, providers=['CPUExecutionProvider'])

    def Generate(self):
        generate(self.togc_session,self.gen_session)


