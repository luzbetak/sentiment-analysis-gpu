#!/usr/bin/env python3
import os

def create_project_structure():
    # Define the structure
    structure = {
        'src': {
            '__init__.py': None,
            'config.py': None,
            'main.py': None,
            'models': {
                '__init__.py': None,
                'classifier.py': None
            },
            'data': {
                '__init__.py': None,
                'dataset.py': None
            },
            'utils': {
                '__init__.py': None,
                'logging_config.py': None,
                'training.py': None
            },
            'data_generation': {
                '__init__.py': None,
                'generate_data.py': None
            }
        }
    }

    def create_structure(base_path, structure):
        for name, contents in structure.items():
            path = os.path.join(base_path, name)
            if contents is None:
                # Create empty file
                open(path, 'a').close()
            else:
                # Create directory and recurse
                os.makedirs(path, exist_ok=True)
                create_structure(path, contents)

    create_structure('.', structure)
    print("Directory structure created successfully!")

if __name__ == "__main__":
    create_project_structure()
