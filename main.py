import supervisely_lib as sly

import cv2
import copy

# Initialize API object
address = 'http://supervisely.private:38585/'
token = 'gvmiNcW8RR1LpatrVaIwrv9mtICOHBS7yhxpaKn70RsCDJZJVk6HpPxtMNhsFcRKkGe7t7DPQueVMG7445FSUUvkuAyAKSEbC3DSh2ek10nv6zJfYfM2NnzV5WMN8EKi'
api = sly.Api(address, token)


"""
STEP 1 — Create project
"""

workspace_id = 354
project_name = 'demo_project'
dataset_name = 'demo_dataset'

project = api.project.create(workspace_id, project_name, type=sly.ProjectType.IMAGES, change_name_if_conflict=True)
dataset = api.dataset.create(project.id, f'{dataset_name}', change_name_if_conflict=True)


# Update project classes
class_defect = sly.ObjClass('defect', sly.Rectangle)
classes = sly.ObjClassCollection([class_defect])
project_meta = sly.ProjectMeta(classes)
updated_meta = api.project.update_meta(project.id, project_meta.to_json())

"""
STEP 1 — Uploade img+ann
"""

# Uploade image
test_image = sly.image.read('./test_image.png') # RGB
height, width = test_image.shape[0], test_image.shape[1]
img_info = api.image.upload_np(dataset.id, name="test_image.png", img=test_image)

# Uploade annotation
label_defect = sly.Label(sly.Rectangle(top=46, left=122, bottom=114, right=180), class_defect)
ann = sly.Annotation((height, width), [label_defect])
api.annotation.upload_ann(img_info.id, ann)


"""
STEP 2 — Downloade and visualize results
"""

# Downloade image
ann_info = api.annotation.download(img_info.id)
ann = sly.Annotation.from_json(ann_info.annotation, project_meta)

# Downloade annotations
img = api.image.download_np(img_info.id)
new_img = copy.deepcopy(img)

# Draw Annotation on image
ann.draw_pretty(img, thickness=1, output_path='./test_image_annotated.png')






