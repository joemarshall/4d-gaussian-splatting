import ast
from pathlib import Path


def list_densifier_options() -> dict[str, list[object]]:
	option_defaults: dict[str, list[object]] = {}
	folder = Path(__file__).resolve().parent.parent / "scene" / "densifiers"
	base_class_name = "DensifierBase"
	base_option_defaults: dict[str, list[object]] = {}
	class_option_defaults: dict[str, dict[str, list[object]]] = {}
	densifier_names: dict[str, str] = {}

	def parse_default_value(call_node: ast.Call) -> object:
		if len(call_node.args) < 2:
			return None

		default_node = call_node.args[1]
		try:
			return ast.literal_eval(default_node)
		except Exception:
			return ast.unparse(default_node)

	def add_default(mapping: dict[str, list[object]], option_name: str, default_value: object) -> None:
		values = mapping.setdefault(option_name, [])
		if default_value not in values:
			values.append(default_value)

	for script_path in sorted(folder.glob("*.py")):
		source = script_path.read_text(encoding="utf-8")
		tree = ast.parse(source, filename=str(script_path))

		for node in tree.body:
			if not isinstance(node, ast.ClassDef):
				continue

			is_base_class = node.name == base_class_name
			is_densifier_subclass = any(
				isinstance(base, ast.Name) and base.id == base_class_name
				for base in node.bases
			)

			if not is_base_class and not is_densifier_subclass:
				continue

			class_defaults = class_option_defaults.setdefault(node.name, {})

			for class_node in ast.walk(node):
				if not isinstance(class_node, ast.Call):
					continue

				if (
					isinstance(class_node.func, ast.Attribute)
					and class_node.func.attr == "_get_option"
					and isinstance(class_node.func.value, ast.Name)
					and class_node.func.value.id == "self"
					and class_node.args
					and isinstance(class_node.args[0], ast.Constant)
					and isinstance(class_node.args[0].value, str)
				):
					option_name = class_node.args[0].value
					default_value = parse_default_value(class_node)
					add_default(option_defaults, option_name, default_value)
					add_default(class_defaults, option_name, default_value)
					if is_base_class:
						add_default(base_option_defaults, option_name, default_value)

				if (
					is_densifier_subclass
					and isinstance(class_node.func, ast.Attribute)
					and class_node.func.attr == "__init__"
					and isinstance(class_node.func.value, ast.Call)
					and isinstance(class_node.func.value.func, ast.Name)
					and class_node.func.value.func.id == "super"
					and len(class_node.args) >= 2
					and isinstance(class_node.args[1], ast.Constant)
					and isinstance(class_node.args[1].value, str)
				):
					densifier_names[node.name] = class_node.args[1].value

	for class_name, densifier_name in densifier_names.items():
		for option_name, default_values in base_option_defaults.items():
			prefixed_option_name = f"{densifier_name}_{option_name}"
			for default_value in default_values:
				add_default(option_defaults, prefixed_option_name, default_value)
		for option_name, default_values in class_option_defaults.get(class_name, {}).items():
			prefixed_option_name = f"{densifier_name}_{option_name}"
			for default_value in default_values:
				add_default(option_defaults, prefixed_option_name, default_value)

	return {option_name: option_defaults[option_name] for option_name in sorted(option_defaults)}


if __name__ == "__main__":
	option_names = list_densifier_options()
	print("Densifier option names:")
	for name, default_values in option_names.items():
		print(f"{name}: {default_values}")