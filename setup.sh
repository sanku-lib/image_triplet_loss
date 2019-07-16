sudo apt-get update
sudo apt-get install python3.6
sudo apt-get install python3-pip
sudo pip3 install -r requirements.txt
echo "Environment Setup Successful."
echo "Downloading Dataset"
python3 download_dataset.py
echo "SETUP DONE"