def create_model(opt):
    model = None
    if opt.model == 'fcrn':
        assert(opt.dataset_mode == 'nyud')
        from .fcrn_model import FCRN_Model
        model = FCRN_Model()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model