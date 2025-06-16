import argparse
from diffusion_model import DiffusionModel
from torchvision.utils import save_image



def parse_arguments():
    """Returns parsed arguments"""
    parser = argparse.ArgumentParser(description="Sample images from diffusion model")
    parser.add_argument("checkpoint_name", type=str, default=None, help="Checkpoint name of diffusion model")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--n-images-per-row", type=int, default=10, help="Number of images each row contains in the grid")
    parser.add_argument("--device", type=str, default=None, help="GPU device to use")
    parser.add_argument("--timesteps", type=int, default=None, help="Total timesteps for sampling")
    parser.add_argument("--beta1", type=float, default=None, help="Hyperparameter for DDPM")
    parser.add_argument("--beta2", type=float, default=None, help="Hyperparameter for DDPM")
    parser.add_argument("--context-image", type=str, default=None, help="Path to unstained slide for paired inference")
    return parser.parse_args()


if __name__=="__main__":    
    args = parse_arguments()
    diffusion_model = DiffusionModel(device=args.device, checkpoint_name=args.checkpoint_name)
    if args.context_image:
        from PIL import Image
        transform, _ = diffusion_model.get_transforms("paired")
        context_img = transform(Image.open(args.context_image).convert('RGB'))
        context_vec = context_img.mean(dim=[1,2]).to(diffusion_model.device)
        samples, _, _ = diffusion_model.sample_ddpm(args.n_samples,
                                                    context_vec,
                                                    args.timesteps,
                                                    args.beta1,
                                                    args.beta2)
        for i, sample in enumerate(samples):
            save_image(sample, f"generated_image_{i}.jpeg")
    else:
        diffusion_model.generate(args.n_samples, args.n_images_per_row, args.timesteps, args.beta1, args.beta2)
