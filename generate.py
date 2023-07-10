from colabcode import *
from pathlib import Path
import argparse

N_SAMPLES = 1024
# MODEL_LOADER = lambda: load_unet_base(1)
# MODEL_LOADER = lambda: load_unet_T(400, 1)
# MODEL_LOADER = lambda: load_unet_noise('beta')
# MODEL_LOADER = lambda: load_unet_T(400, 2)
# MODEL_LOADER = lambda: load_unet_T(400, 3)
MODEL_LOADER = lambda: load_unet_T(400, 2)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", action='store_true')
  parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
  parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
  args = parser.parse_args()

  with torch.no_grad():
    print("Loading model")
    model = MODEL_LOADER()
    model.denoiser.eval()

    print("Generating samples")
    if args.test:
      print("Test run, generating 3 samples")
      samples = generate_samples(model, 3, batch_size=1)
    else:
      samples = generate_samples(model, args.n_samples, batch_size=args.batch_size)

    print("Saving images")
    Path("samples").mkdir(parents=True, exist_ok=True)
    resize_op = torchvision.transforms.Resize((28, 28))
    for i_img, img in enumerate(samples):
      img = resize_op(img)
      torchvision.utils.save_image(img, f'samples/{i_img}.png')



    


