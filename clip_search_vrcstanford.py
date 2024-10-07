import os
import random
import gradio as gr
import torch
import clip
import numpy as np
import pandas as pd

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print('Using ' + device)

features_path = 'features/'



photo_features = np.load(features_path + "features.npy")
photo_ids = pd.read_csv(features_path + "photo_ids.csv")
photo_ids = list(photo_ids['photo_id'])


def clip_search(search_string):
        
    with torch.no_grad():
        # Encode and normalize the description using CLIP
        text_encoded = model.encode_text(clip.tokenize(search_string).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        # Retrieve the description vector and the photo vectors
    text_features = text_encoded.cpu().numpy()

    # Compute the similarity between the descrption and each photo using the Cosine similarity
    similarities = list((text_features @ photo_features_wh.T).squeeze(0))

    # Sort the photos by their similarity score
    candidates = sorted(zip(similarities, range(photo_features_wh.shape[0])), key=lambda x: x[0], reverse=True)
    
    images = []
    for i in range(60):
        # Retrieve the photo ID
        idx = candidates[i][1]
        photo_id = photo_ids_wh[idx]
        images.append([('images/' + str(photo_id) + '.jpg'),  photo_id])
#         images.append([('plimages/photogrammar/' + photo_id + '.jpg'), 'https://photogrammar.org/photo/' + photo_id + '/PP'])
    
#     print(images)
    return images

css = "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"
with gr.Blocks(css = css) as demo:
    with gr.Column(variant="panel"):
        with gr.Row(variant="compact"):
            search_string = gr.Textbox(
                label="Evocative Search",
                show_label=True,
                max_lines=1,
                placeholder="Type something abstruse, or click a suggested search below.",
            ).style(
                container=False,
            )
            btn = gr.Button("Retrieve Images", variant="primary").style(full_width=False)
        with gr.Row(variant="compact"):
            suggest1 = gr.Button("rococo", variant="secondary").style(size="sm")    
            suggest2 = gr.Button("brutalism", variant="secondary").style(size="sm")    
            suggest3 = gr.Button("classical", variant="secondary").style(size="sm")       
            suggest4 = gr.Button("gothic", variant="secondary").style(size="sm")    
            suggest5 = gr.Button("eating together", variant="secondary").style(size="sm")   
        gallery = gr.Gallery(
            label=False, show_label=False, elem_id="gallery"
        ).style(grid=[6], height="100%",)

    suggest1.click(clip_search, inputs=suggest1, outputs=gallery)
    suggest2.click(clip_search, inputs=suggest2, outputs=gallery)
    suggest3.click(clip_search, inputs=suggest3, outputs=gallery)
    suggest4.click(clip_search, inputs=suggest4, outputs=gallery)
    suggest5.click(clip_search, inputs=suggest5, outputs=gallery)
    btn.click(clip_search, inputs=search_string, outputs=gallery)
    search_string.submit(clip_search, search_string, gallery)



if __name__ == "__main__":
    demo.launch(share=False, server_name='0.0.0.0', server_port=7860)
    demo.close()
#     demo.launch()
