import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

# this is probably a bad design to import this file only to run following:
jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)

