import pretrainedmodels

print(pretrainedmodels.model_names)
print(pretrainedmodels.pretrained_settings['nasnetalarge'])
model_name = 'nasnetalarge' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()