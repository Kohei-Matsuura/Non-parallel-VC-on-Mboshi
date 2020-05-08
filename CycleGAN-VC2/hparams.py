"""
This is hyper-parameters of models.
When used, it have to be imported.
"""

hd = {}

# Settings
hd['align3'] = '{:-^30}'.format('Setting')
hd['BATCH_SIZE'] = 5
hd['WIDTH'] = 128

hd['align4'] = '{:-^25}'.format('')
hd['STEP_NUM'] = int(1e5)
hd['IDT_STOP_STEP'] = int(1e4)
hd['SAVE_PERIOD'] = 5e3
hd['FEATURE_SIZE'] = 40
hd['Lambda_idt'] = 5.0
hd['Lambda_cyc'] = 10.0

hd['align7'] = '{:-^25}'.format('')
# Never changed (maybe)
hd['SEED'] = 100
hd['LEARNING_RATE_G'] = 2e-4
hd['LEARNING_RATE_D'] = 1e-4
hd['WEIGHT_DECAY'] = 0.0 #1e-5
hd['CLIPPING'] = 5.0
#hd['DROPOUT'] = 0.1

hd['align10'] = '{:-^30}'.format('')
