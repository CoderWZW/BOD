from SELFRec import SELFRec
from util.conf import ModelConf

if __name__ == '__main__':
    baseline = ['NCF','LightGCN','MF','NGCF','DirectAU']
    graph_models = ['SGL', 'SimGCL']
    data_augmentation = ['BOD']

    print('Baseline Models:')
    print('   '.join(baseline))
    print('-' * 80)
    print('Graph-Based Models:')
    print('   '.join(graph_models))
    print('-' * 80)
    print('Denoising Models:')
    print('   '.join(data_augmentation))

    print('=' * 80)
    model = input('Please enter the model you want to run:')
    import time

    s = time.time()
    if model in baseline or model in graph_models or model in data_augmentation:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
