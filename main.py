import BlahutArimoto as BA
from BernoulliCommon import BernoulliObject as Ber
from BernoulliTypesCommon import BernoulliTypesObject as BerTypes
import PlotCommon as PC
import torch as tr
import math
import argparse

def whole_BA(output_size: int, d_threshold: float = 0.0002, d_step:float = 0.00001, use_cuda: bool = True):

    initial_input_size = math.floor(output_size ** (63/100))

    # BerObj = Ber(initial_input_size, output_size)
    BerObj = BerTypes(initial_input_size, output_size)
    C, r, dr = BA.blahut_arimoto_derivative(BerObj, d_threshold, d_step, use_cuda=use_cuda)

    print("[+] Checking one more delta")
    one_more_delta, C_new, r_new, dr_new = BA.check_if_one_more_delta(BerObj, C=C[-1], d_threshold=d_threshold, d_step=d_step, use_cuda=use_cuda)
    if one_more_delta:
        print("[+] Added delta: now has {} deltas".format(BerObj.input_size))
        return BerObj.theta_vector, C_new, r_new, dr_new
    print("[+] Did not add delta: stayed with {} deltas".format(BerObj.input_size))
    return BerObj.theta_vector, C, r, dr


def naive_BA(input_size: int, output_size: int, conditional: bool, use_cuda: bool):

    # BerObj = Ber(input_size, output_size)
    BerObj = BerTypes(input_size, output_size)

    # theta_vector_helper = BerObj.theta_vector + [0.5 - 0.01, 0.5 + 0.01]
    # odd = True if BerObj.input_size % 2 == 1 else False
    # if odd == False:
    #     theta_vector_helper += [0.5]
    # theta_vector_helper.sort()

    # BerObj.update_input_size(len(theta_vector_helper), theta_vector_helper)

    dPyx = BerObj.update_Pyx_get_dPyx()

    if conditional:
        BerObj.update_Py_plus_1_x()

    C, r = BA.blahut_arimoto(BerObj, conditional=conditional, use_cuda=use_cuda)

    dr = BerObj.get_dr(r, dPyx)

    return BerObj.theta_vector, C, r, dr

def derivative_BA(input_size: int, output_size: int, d_threshold: float = 0.02, d_step:float = 0.0001, use_cuda: bool = True):

    # BerObj = Ber(input_size, output_size)
    BerObj = BerTypes(input_size, output_size)

    C, r, dr = BA.blahut_arimoto_derivative(BerObj, d_threshold, d_step, use_cuda=use_cuda)
    return BerObj.theta_vector, C, r, dr

def main():

    parser = argparse.ArgumentParser(description='Calaulate the Capacity and r(theta) using Blahut-Arimoto algorithm')
    parser.add_argument('-o', '--output_size', help='Set the out size in bits')
    parser.add_argument('-i', '--input_size', help='Set the input size in bits')
    parser.add_argument('-n', '--naive_BA', action='store_true', help='Use the naive Blahut-Arimoto algorithm. [!] Can only be used with output size')
    parser.add_argument('-c', '--conditional', action='store_true', help='Use conditional Blahut-Arimoto calculation')
    parser.add_argument('--use_cuda', action='store_true', help='Use gpu instead of gpu')

    args = parser.parse_args()

    output_size = 13
    input_size = 11
    # args.input_size = None
    # args.use_cuda = True

    if args.output_size:
        output_size = int(args.output_size)

    if args.input_size:
        input_size = int(args.input_size)
        if args.naive_BA:
            theta_vector, C, r, dr = naive_BA(input_size, output_size, conditional=args.conditional, use_cuda=args.use_cuda)
        else:
            theta_vector, C, r, dr = derivative_BA(input_size, output_size, use_cuda=args.use_cuda)
    else:
        theta_vector, C, r, dr = whole_BA(output_size, use_cuda=args.use_cuda)

    # print("[+] output_size    # of deltas ")
    # for i in range(4, 20):
    #     theta_vector, C, r, dr = whole_BA(i, d_threshold = 0.02, d_step= 0.0001, use_cuda=args.use_cuda)
    #     print("[+]       {}              {}      ".format(i, len(theta_vector)))
    # exit()

    # PC.plot_r(theta_vector, output_size, r)
    PC.plot_r_dr_C(theta_vector, output_size, r, dr, C)

    # output_size = 1000
    # theta_vector = [0, 0.0024878947368424777, 0.007595789473684972, 0.014783684210524468, 0.02393157894736217, 0.03502947368420338, 0.04791736842105453, 0.0625152631579062, 0.07864315789474716, 0.0961010526315876, 0.11438894736841859, 0.13283684210525126, 0.15067473684209803, 0.16752263157894381, 0.18332052631578855, 0.19816842105263233, 0.2123163157894754, 0.2258642105263179, 0.23907210526316003, 0.25194000000000183, 0.26470789473684353, 0.27740578947368516, 0.29008368421052677, 0.30283157894736845, 0.3156394736842102, 0.3285973684210521, 0.341595263157894, 0.3547631578947361, 0.3679110526315782, 0.38122894736842045, 0.3944368421052626, 0.4078247368421049, 0.420992631578947, 0.43430052631578925, 0.44744842105263133, 0.46064631578947346, 0.4737742105263155, 0.4869021052631576, 0.4999999999999996, 0.513097894736842, 0.5262257894736843, 0.5393536842105265, 0.5525515789473685, 0.5656994736842107, 0.5790073684210522, 0.5921752631578943, 0.6055631578947354, 0.6187710526315773, 0.6320889473684187, 0.6452368421052609, 0.658404736842103, 0.6714026315789459, 0.6843605263157889, 0.6971684210526327, 0.7099163157894767, 0.722594210526321, 0.7352921052631652, 0.7480600000000092, 0.7609278947368526, 0.7741357894736945, 0.7876836842105349, 0.8018315789473726, 0.816679473684207, 0.8324773684210371, 0.8493252631578625, 0.8671631578946833, 0.8856110526315014, 0.9038989473683202, 0.9213568421051428, 0.9374847368419714, 0.952082631578807, 0.9649705263156504, 0.9760684210525019, 0.9852163157893623, 0.9924042105262316, 0.9975121052631104, 1]
    # r = None
    # C = None

    import time
    from pathlib import Path
    t = time.localtime()
    input_x_output = '{}x{}_'.format(len(theta_vector), output_size)
    file_name = time.strftime("%H-%M-%S_%d-%m-%Y.pt", t)
    file_obj = {'theta': theta_vector, 'r': r, 'C': C}

    path = './outputs/' #r'D:/User/University/MSc/final_project/files/'
    tr.save(file_obj, path+input_x_output+file_name)

if __name__ == "__main__":
    main()
