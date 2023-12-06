from diffusers import StableDiffusionPipeline
import torch
import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from torch.cuda.amp import autocast  # Import autocast from the correct module

app = tk.Tk()
app.geometry("542x642")
app.title("Image Generation App")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, height=40, width=512, text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise use CPU
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token="your_token")

def generate():
    with autocast("cuda"):  # Use "cuda" instead of the device variable
        image = pipe(prompt.get(), guidance_scale=8.5).images[0]
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

trigger = ctk.CTkButton(app, height=40, width=120, text_color="white", fg_color="green", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()  # Add the mainloop to run the Tkinter application



app.mainloop()