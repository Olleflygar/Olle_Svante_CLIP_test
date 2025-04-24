import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from clip import clip
import numpy as np
from torch.nn import functional as F

### Loading dataset ###
def main():
    N = 70

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(3)])
    mnist = datasets.MNIST(root = './data', download = True, train = True, transform=transform)
    
    images = [mnist[i][0] for i in range(N)]
    labels = [str(mnist[i][1]) for i in range(10)]
    prompts = [f'A photo of the handwritten digit {i}' for i in labels]

    ### CLIP encoding ###
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, preprocess = clip.load('ViT-B/32', device = device)
    print('Preprocesssing...')
    #preprocess images/prompts before encoding
    image_input = torch.stack([preprocess(img) for img in images]).to(device)
    text_input = clip.tokenize(prompts).to(device)

    ### Encoding ###
    print('Computing embeddings...')
    BATCH_SIZE = 200  # Adjust based on memory (start small: 32, 64...)

    all_image_features = []

    with torch.no_grad():
        for i in range(0, len(image_input), BATCH_SIZE):
            batch = image_input[i:i+BATCH_SIZE].to(device)
            features = model.encode_image(batch)
            all_image_features.append(features.cpu())

    enc_images = torch.cat(all_image_features, dim=0)  # Final shape: (N, D)

    with torch.no_grad():
        enc_prompts = model.encode_text(text_input)
    
    print('Done with embedding computation')
    ### Normalization ###



    enc_prompts = F.normalize(enc_prompts, dim = -1)
    enc_images = F.normalize(enc_images, dim = -1)
    labels = [mnist[i][1] for i in range(N)]

    torch.save({'image': enc_images,
                'labels': labels
                }, 'image_embeddings.pt')

    torch.save({'prompt': enc_prompts,
                'labels': labels
                }, 'text_embeddings.pt')
    print('Embeddings saved!')
    print('Job finished')

if __name__ == '__main__':
    main()
