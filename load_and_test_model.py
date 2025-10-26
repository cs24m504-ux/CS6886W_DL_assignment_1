import argparse
import torch
import os
import sys

# Import helper functions and constants from the sweep script
try:
	from vgg6_sweep import GetCifar10, DEVICE, vgg
	# vgg6_sweep defines an `eval` function; import it under a different name to avoid shadowing built-in eval
	from vgg6_sweep import eval as evaluate
except Exception as e:
	print(f"Failed to import helpers from vgg6_sweep.py: {e}")
	raise


def load_model(path, device):
	"""Load a model checkpoint. Handles full-model and state-dict formats."""
	if not os.path.exists(path):
		raise FileNotFoundError(f"Model file not found: {path}")

	# Try loading to the target device
	try:
		loaded = torch.load(path, map_location=device)
	except TypeError:
		# Older torch versions may not accept device object; try string
		loaded = torch.load(path, map_location=str(device))

	# If the file contains a full nn.Module, return it
	if isinstance(loaded, torch.nn.Module):
		model = loaded
	elif isinstance(loaded, dict):
		# Assume it's a state_dict; construct a VGG6 model and load the weights
		cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']
		model = vgg(cfg_vgg6, num_classes=10, batch_norm=True, activation_function='ReLU')
		try:
			model.load_state_dict(loaded)
		except Exception:
			# Sometimes the checkpoint is saved as {'model_state_dict': state_dict, ...}
			if 'model_state_dict' in loaded:
				model.load_state_dict(loaded['model_state_dict'])
			elif 'state_dict' in loaded:
				model.load_state_dict(loaded['state_dict'])
			else:
				raise
	else:
		raise RuntimeError(f"Unrecognized checkpoint format: {type(loaded)}")

	model.to(device)
	model.eval()
	return model


def main():
	parser = argparse.ArgumentParser(description="Load a trained model and evaluate on CIFAR-10 test set")
	parser.add_argument("--model", "-m", default="balmy-sweep-9_model.pth", help="Path to the model file to load")
	parser.add_argument("--batch-size", "-b", type=int, default=256, help="Batch size for evaluation")
	args = parser.parse_args()

	model_path = args.model
	batch_size = args.batch_size

	try:
		model = load_model(model_path, DEVICE)
	except Exception as e:
		print(f"Error loading model: {e}")
		sys.exit(1)

	# Prepare test loader
	try:
		_, _, test_loader = GetCifar10(batch_size)
	except Exception as e:
		print(f"Failed to create data loaders: {e}")
		sys.exit(1)

	# Evaluate
	try:
		test_acc = evaluate(model, test_loader)
	except Exception as e:
		print(f"Evaluation failed: {e}")
		sys.exit(1)

	print(f"Test accuracy of loaded model ({os.path.basename(model_path)}): {test_acc:.2f}%")


if __name__ == "__main__":
	main()
	
# How to run:
# Command format: 
#   python load_model_from_file.py --model <path/to/your/model.pth> --batch-size <batch_size>
# Example: 
#   python .\load_model_from_file.py --model .\models\wise-sweep-9_model.pth --batch-size 256