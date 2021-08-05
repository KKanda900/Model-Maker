##########################################################################
# Note the instructions are for Windows Users ONLY                       #

# For any other OS the installation might differ                         #
##########################################################################

import os

# Install the main libraries
def install_libraries():
    os.system("pip install opencv-python")
    os.system("pip install Pillow")
    os.system("pip install PySimpleGUI")
    os.system("pip install python-tk")
    os.system("pip install tensorflow")
    os.system("pip install pandas")
    os.system("pip install matplotlib")
    os.system("pip install sklearn")
    os.system("pip install pyinstaller")

# Install both the Application and the Model Creator Application
def install_applications():
    os.system("pyinstaller --noconfirm --onefile --windowed {}".format("./Model_Creator_App.py"))

# Upon starting the setup, give options to the user to pick from
if __name__ == "__main__":
    # iterate until the user is done
    while True:
        # get the user's input choice
        user_input = input("Would you like to install the applications, install the libraries or quit? (app) (lib) (quit): ")
        # install the application if they choose app
        if user_input == "app":
            install_applications()
        # install the libraries if they chose lib
        elif user_input == "lib":
            install_libraries()
        # otherwise just quit the application
        else:
            break