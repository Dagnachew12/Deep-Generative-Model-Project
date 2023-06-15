from Train import *
from Eval import *
if __name__ == '__main__':
    main()
    save_examples(ct_genn, mri_genn, test_loader, 'Data/saved_img/test')
