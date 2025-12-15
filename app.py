import os
from io import BytesIO
from zipfile import ZipFile

import aiofiles
import cv2
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from remove_bg import remove_background_with_bboxes

from ultralytics import YOLO

app = FastAPI()

origins = ["*"]

model = YOLO('models/wood-detection-model.pt')
model.export(format='openvino')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def zip_files(files):
    zip_subdir = "archive"
    zip_filename = "%s.zip" % zip_subdir

    s = BytesIO()
    zf = ZipFile(s, "w")

    for fpath in files:
        fdir, fname = os.path.split(fpath)
        zip_path = os.path.join(zip_subdir, fname)
        zf.write(fpath, zip_path)

    zf.close()

    resp = Response(s.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })

    return resp


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/image-upload")
async def upload_image(images: list[UploadFile]):
    images_to_send = []
    uploads_counter = len([i for i in os.listdir("uploads")])
    first_upload_index = uploads_counter
    curr_upload_name = f"uploads/upload_{uploads_counter}.jpg"
    for image in images:
        async with aiofiles.open(curr_upload_name, 'wb+') as f:
            content = await image.read()
            await f.write(content)
        uploads_counter += 1
        curr_upload_name = f"uploads/upload_{uploads_counter}.jpg"
    print(len(images))
    for i in range(len(images)):
        curr_upload_name = f"uploads/upload_{first_upload_index + i}.jpg"
        res = model.predict(
            source=curr_upload_name,
            conf=0.5,
            save=False,
            device='cpu'
        )
        res[0].save(f"bgs/upload_{first_upload_index + i}.jpg")
        result_img = remove_background_with_bboxes(res[0])
        no_bg_name = f"predictions/upload_{first_upload_index + i}_no_bg.jpg"
        cv2.imwrite(no_bg_name, result_img)
        images_to_send.append(curr_upload_name.replace('uploads', 'bgs/'))
        images_to_send.append(no_bg_name)
    print(images_to_send)
    return zip_files(images_to_send)
