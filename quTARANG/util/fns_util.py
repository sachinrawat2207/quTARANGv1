import os

# Return the last saved file 
def get_last_saved_file(directory):
    files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    last_saved_file = max(files, key=lambda file: os.path.getmtime(os.path.join(directory, file)))
    return last_saved_file

# Use to avoid overwriting of energy and rms file if file already exist. this is used when program is resumed.
def new_filename(file_name):
    base_name, extension = os.path.splitext(file_name)
    counter = 1

    while os.path.exists(file_name):
        file_name = f"{base_name}_{counter}{extension}"
        counter += 1
    return file_name

