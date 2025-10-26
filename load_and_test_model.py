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
	except Exception as e:
		# Handle common pickle issue when a model was saved from a script where
		# the model class lived in __main__ (e.g. saved from a script run) and
		# is now being loaded from a different module. The unpickler then
		# cannot find the class in __main__ and raises "Can't get attribute ...".
		msg = str(e)
		if "Can't get attribute" in msg or "can't get attribute" in msg:
			# Try to make the VGG class available under __main__ so the
			# pickle can resolve it. Import vgg6_sweep and inject the class
			# into the current __main__ module under the missing name.
			import re
			import vgg6_sweep as sweep_mod
			import sys as _sys

			m = re.search(r"Can't get attribute '(.+?)' on", msg)
			attr_name = m.group(1) if m else 'VGG'
			main_mod = _sys.modules.get('__main__')
			injected = False
			try:
				if main_mod is not None and hasattr(sweep_mod, attr_name):
					setattr(main_mod, attr_name, getattr(sweep_mod, attr_name))
					injected = True
				elif main_mod is not None and hasattr(sweep_mod, 'VGG'):
					# Fallback: inject VGG specifically
					setattr(main_mod, 'VGG', sweep_mod.VGG)
					injected = True

				if injected:
					try:
						loaded = torch.load(path, map_location=device)
					finally:
						# cleanup injected attribute to avoid side-effects
						try:
							if main_mod is not None and hasattr(main_mod, attr_name):
								delattr(main_mod, attr_name)
						except Exception:
							pass
				else:
					raise
			except Exception:
				# Re-raise the original error if injection/loading failed
				raise e
		else:
			# Re-raise if it's some other error
			raise

	# If the file contains a full nn.Module, return it
	if isinstance(loaded, torch.nn.Module):
		print("Loaded full model from checkpoint.")
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


def save_model_state_dict(model, out_path):
	"""Save an nn.Module's state_dict to out_path.

	This helper always saves model.state_dict() so the resulting file is
	portable and can be loaded into an identical model definition later.
	"""
	if not isinstance(model, torch.nn.Module):
		raise TypeError("model must be an instance of torch.nn.Module")

	dirpath = os.path.dirname(out_path)
	if dirpath and not os.path.exists(dirpath):
		os.makedirs(dirpath, exist_ok=True)

	torch.save(model.state_dict(), out_path)
	print(f"Saved state_dict to: {out_path}")


def main():
	parser = argparse.ArgumentParser(description="Load a trained model and evaluate on CIFAR-10 test set")
	parser.add_argument("--model", "-m", default="balmy-sweep-9_model.pth", help="Path to the model file to load")
	parser.add_argument("--batch-size", "-b", type=int, default=64, help="Batch size for evaluation")
	parser.add_argument("--save-state", action="store_true", help="Save the loaded model's state_dict for portability")
	parser.add_argument("--save-path", default=None, help="Optional path to write the state_dict (overrides default naming)")
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

	print(f"\nTest accuracy of loaded model ({os.path.basename(model_path)}): {test_acc:.2f}%")

	# Optionally save state_dict for portability
	if args.save_state:
		# Determine output path: use provided save path or derive from the model file name
		out_path = args.save_path
		if not out_path:
			base, ext = os.path.splitext(model_path)
			out_path = base + "_state_dict.pth"
		try:
			save_model_state_dict(model, out_path)
		except Exception as e:
			print(f"Failed to save state_dict: {e}")
			# Do not treat this as a hard failure for the evaluation run


if __name__ == "__main__":
	main()
	
# How to run:
# Command format: 
#   python load_model_from_file.py --model <path/to/your/model.pth> --batch-size <batch_size>
# Example: 
#   python .\load_model_from_file.py --model .\models\wise-sweep-9_model.pth --batch-size 64