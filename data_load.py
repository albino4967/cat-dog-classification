import gdown

file_destinations = {'CatDogClassification':'cat-dog-dataset.zip'}

file_id_dic = {'CatDogClassification':'1Z9JUrCDGaAJBGfmCw2BjM0qY-G9xLhLm'}

def download_file_from_google_drive(id_, destination):
    url = f'https://drive.google.com/uc?id={id_}'
    output = destination
    gdown.download(url, output, quiet=False)
    print(f'{output} download complete!')

def main():
    download_file_from_google_drive(id_=file_id_dic['CatDogClassification'], destination=file_destinations['CatDogClassification'])

if __name__ == "__main__" :
    main()