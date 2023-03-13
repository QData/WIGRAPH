
def get_orig_model_outputs(inputs, model):
  orig_output, orig_embedding = model(inputs, flag = 'test', mode = False, interpret_mode='lime')
  orig_output = orig_output.detach()
  orig_embedding = orig_embedding.detach()
  return orig_output, orig_embedding
