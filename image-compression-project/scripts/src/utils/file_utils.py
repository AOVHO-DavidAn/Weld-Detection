def create_directory(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def copy_file(src, dst):
    import shutil
    shutil.copy(src, dst)

def get_new_image_path(original_path, output_dir):
    import os
    base_name = os.path.basename(original_path)
    new_name = os.path.splitext(base_name)[0] + '.jpg'
    return os.path.join(output_dir, new_name)

def get_new_json_path(original_path, output_dir):
    import os
    base_name = os.path.basename(original_path)
    new_name = os.path.splitext(base_name)[0] + '.json'
    return os.path.join(output_dir, new_name)