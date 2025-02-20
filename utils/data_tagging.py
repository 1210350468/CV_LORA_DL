# import clip
import torch
from PIL import Image
import os
import csv

# 加载CLIP模型
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# 定义一组标签词语，带有文件夹名称
labels_dict = {
    "AnnualCrop": [
        'satellite image',"field", "agriculture", "crop", "farming", "plants", "farmland",
    ],
    "Forest": [
        'satellite image',"natural", "environment", "dense tree coverage"
    ],
    "HerbaceousVegetation": [
        'satellite image',"vegetation", "herbs", "plants", "greenery", "natural", "field", "dense", "wild", "grassland"
    ],
    "Highway": [
        'satellite image',"urban areas", "rural landscapes", "overpasses", "bridges"
    ],

    "Industrial": [
        'satellite image',"factories", "warehouses", "large rectangular structures",
    ],
    "Pasture": [
        'satellite image',"grass", "dense grass in well-maintained areas", "sparse grass in overgrazed areas"
    ],
    "PermanentCrop": [
        'satellite image',"apple orchards", "orange tree orchards", "grape vineyards"
    ],
    "Residential": [
        'satellite image',"individual houses", "multi-story apartment buildings"
    ],
    "River": [
        'satellite image',"shades of blue", "green water", "clear water sections"
    ],

    "SeaLake": [
        'satellite image',"shades of blue", "green water in certain conditions", "clear water areas"
],


}
# labels_dict = {
#     "AnnualCrop": [
#         'satellite image', "AnnualCrop","field", "agriculture", "crop", "farming", "plants", "farmland", 
#         "rural", "green", "seasonal", "grain crops", "vegetable crops", 
#         "light green early growth", "dark green late summer", 
#         "yellow before harvest", "brown before harvest", 
#         "straight lines in field", "curved patterns in field", 
#         "drip irrigation", "sprinkler systems", "small field", "large field", 
#         "clear row structures", "distinct rows", "sparse crops soil", 
#         "soil between rows", "mechanized impact", "adjacent to forests", 
#         "near urban areas", "high resolution", "low resolution", 
#         "early spring color", "mid-summer color", "late summer color", 
#         "brownish drought color", "beige post-harvest color", 
#         "fine young crop texture", "coarse mature crop texture", 
#         "straight line texture", "curved line texture", 
#         "wet soil texture", "cracked soil texture", "rough harvested field texture", 
#         "uneven harvested field texture", "light green spring color", 
#         "dark green late summer color", "dense meadows", "sparse grasslands", 
#         "cultivated fields", "natural grasslands", "cereal crops land use", 
#         "fodder crops land use", "tractor trails", "fenced areas", 
#         "yellowish autumn color", "barren winter color", 
#         "flood irrigation", "drip line patterns", "smooth grass texture", 
#         "rough weed texture", "straight plowed lines", "irregular natural edges", 
#         "legume crops", "cereal grain fields", "soil in overgrazed areas", 
#         "soil patches in tilled fields", "border forests", "adjacent to water bodies", 
#         "individual plants high resolution", "field shapes low resolution", 
#         "mowed patterns", "baled hay", "early sprouting stages", 
#         "pre-harvest maturity", "circular sprinkler patterns", "linear irrigation tracks", 
#         "small family plots", "large commercial farms"
#     ],
#     "Forest": [
#         'satellite image',"Forest","trees", "woodland", "forest", "nature", "greenery", "foliage", 
#         "dense", "natural", "environment", "dense tree coverage", 
#         "deciduous trees", "evergreen trees", "uniform canopy structure", 
#         "irregular canopy structure", "bright green spring color", 
#         "deep green summer color", "orange autumn color", "red autumn color", 
#         "muted green winter color", "bare branches winter color", 
#         "brownish drought color", "vibrant green healthy vegetation", 
#         "intricate leaf texture", "varied leaf texture", 
#         "complex branch texture", "mosaic canopy texture", 
#         "shrubs ground cover texture", "grass ground cover texture", 
#         "rough bark texture", "smooth bark texture", "forest floor leaves texture", 
#         "forest floor soil texture", "morning shadows", "overcast shadows", 
#         "forests on slopes", "trees in valleys", "logging indicators", 
#         "visible roads", "high spatial resolution", "low spatial resolution"
#     ],
#     "HerbaceousVegetation": [
#         'satellite image',"HerbaceousVegetation", "vegetation", "herbs", "plants", "greenery", "natural", 
#         "field", "dense", "wild", "grassland", "light green in spring", 
#         "dark green in late summer", "densely packed meadows", "sparse grasslands", 
#         "cultivated fields", "natural grasslands", "cereal crops land use", 
#         "fodder crops land use", "tractor trails", "fenced areas", 
#         "yellowish autumn color", "barren winter color", 
#         "flood irrigation", "drip line patterns", "smooth grass texture", 
#         "rough weed texture", "straight plowed lines", "irregular natural edges", 
#         "legume crops", "cereal grain fields", "soil in overgrazed areas", 
#         "soil patches in tilled fields", "border forests", "adjacent to water bodies", 
#         "individual plants high resolution", "field shapes low resolution", 
#         "mowed patterns", "baled hay", "early sprouting stages", 
#         "pre-harvest maturity", "circular sprinkler patterns", "linear irrigation tracks", 
#         "small family plots", "large commercial farms"
#     ],
#     "Highway": [
#         'satellite image',"Highway", "paved roads", "lane markings", "moving cars", "moving trucks", 
#     "grassy areas", "surrounded by trees", "road signs", "safety barriers", 
#     "cloverleaf junctions", "cross intersections", "busy traffic", "sparse traffic", 
#     "two lanes", "multiple lanes", "asphalt surface", "concrete surface", 
#     "urban areas", "rural landscapes", "overpasses", "bridges", 
#     "tunnel openings", "rest stops", "parking facilities", 
#     "detailed road surface", "clear road markings", "highway layout", 
#     "construction areas", "repair sections", "guardrails", 
#     "median barriers", "streetlights", "crosswalks", "traffic lights"
#     ],

#     "Industrial": [
#         'satellite image', "Industrial","factories", "warehouses", "large rectangular structures", "gray colors", 
#     "brown colors", "tall chimneys", "round storage silos", "trucks", 
#     "forklifts", "railway tracks", "large parking areas", "minimal green spaces", 
#     "wide roads", "outdoor storage of materials", "ongoing construction", 
#     "cooling towers", "visible piping systems", "shipping containers", 
#     "discolored ground", "discolored water", "perimeter fencing", 
#     "flat roofs", "building details", "industrial complex outlines", 
#     "power generation plants", "waste disposal areas", "near rivers", 
#     "near lakes", "designated parking lots", "canteens", "separated from residential areas"
#     ],
#     "Pasture": [
#         'satellite image', "Pasture","grass", "dense grass in well-maintained areas", "sparse grass in overgrazed areas",
#     "vibrant green in spring", "rich green in summer", "yellowish hues in autumn",
#     "brown tones during droughts", "grazed by cattle and sheep", 
#     "wooden fences", "wire fences", "ponds", "small lakes", 
#     "sporadic clusters of trees", "solitary trees", "irregular boundaries",
#     "small to large fields", "barns", "sheds", "dirt tracks", "small roads",
#     "borders forests", "borders croplands", "bare patches", "erosion",
#     "flat terrain", "rolling hills", "sprinklers", "water troughs",
#     "detailed vegetation texture", "general layout discernible",
#     "growth in spring and summer", "dormancy in winter", "areas mowed",
#     "signs of reseeding", "small pastures", "large grazing areas",
#     "bordering forests", "adjacent to croplands", "situated on flat terrain",
#     "located on gently rolling hills"
#     ],
#     "PermanentCrop": [
#         'satellite image',"PermanentCrop", "apple orchards", "orange tree orchards", "grape vineyards",
#     "straight crop rows", "curved row patterns", "green shades in growing season",
#     "brown in dormant season", "yellow in dormant season", 
#     "uniformly spaced trees", "varied tree spacing", "drip irrigation", 
#     "sprinkler irrigation", "soil between rows", "soil around plants", 
#     "small plots", "large farms", "next to other farms", 
#     "close to natural vegetation", "processing buildings", "storage facilities", 
#     "vehicle access roads", "fences", "natural barriers like hedges", 
#     "near ponds", "near rivers", "detail visible in plants", 
#     "overall layout discernible", "harvest activity", "pruning in dormant periods",
#     "areas under maintenance", "beekeeping equipment"
#     ],
#     "Residential": [
#         'satellite image',"Residential", "individual houses", "multi-story apartment buildings",
#     "clustered housing arrangements", "spaced-out individual homes",
#     "flat roofs", "sloped roofs", "white exteriors", "red exteriors", 
#     "brick exteriors", "surrounded by gardens", "encircled by lawns", 
#     "include driveways", "parked cars", "grid street layout", 
#     "curved street layout", "irregular street layout", 
#     "nearby parks", "green spaces", "playgrounds in the vicinity", 
#     "private swimming pools", "communal swimming pools", 
#     "fenced properties", "gated community borders", 
#     "detailed building structures visible", "overall housing layout discernible", 
#     "near main roads", "away from busy roads", 
#     "surrounded by trees", "enclosed by shrubs", 
#     "diverse landscaping styles", "pedestrian walkways"
#     ],
#     "River": [
    #     'satellite image',"River", "shades of blue", "green water", "clear water sections", 
    # "turbid or murky water", "narrow rivers", "wide rivers", 
    # "dense vegetation", "sparse vegetation", 
    # "border agricultural lands", "flank urban areas", 
    # "bridges crossing", "natural meandering courses", 
    # "artificially straightened sections", "small islands", 
    # "large islands", "river deltas", "tributary confluences", 
    # "visible floodplains", "areas with clear water", 
    # "areas with murky water", "boats", "ships", 
    # "near parks or recreational areas", 
    # "detailed water textures visible", "general river course discernible", 
    # "well-maintained riparian areas", "degraded riparian zones", 
    # "dams", "weirs", "water intake facilities", 
    # "shades of deep blue"
#     ],

#     "SeaLake": [
#         'satellite image',"SeaLake","shades of blue", "green water in certain conditions", 
#     "clear water areas", "turbid water areas", 
#     "murky water areas", "sandy beaches", 
#     "rocky shorelines", "pebble-strewn shorelines", 
#     "vegetated shores", "non-vegetated shores", 
#     "piers", "docks", "marinas", 
#     "small islands", "large islands", 
#     "border agricultural areas", "near urban areas", 
#     "small boats", "large ships", 
#     "recreational beaches", "adjacent to parks", 
#     "aquaculture areas", "pollution signs", 
#     "detailed shorelines and water textures visible", 
#     "general sea or lake layout discernible", 
#     "wave patterns", "calm water in sheltered areas", 
#     "tidal flats", "algal blooms", 
#     "visible water currents", "ice or snow cover"
# ],
# # 根据图片前缀生成标签
# def generate_labels_for_image(image_name):
#     prefix = image_name.split('_')[0]
#     if prefix in labels_dict:
#         return f"{', '.join(labels_dict[prefix])}, {prefix}"
#     return f"Unknown Category, {prefix}"

# def batch_annotate_and_save(image_dir, output_csv):
#     # 打开CSV文件准备写入
#     output_csv_path=os.path.join(image_dir, output_csv)
#     with open(output_csv_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         # 写入CSV头
#         writer.writerow(['file_name', 'text'])
        
#         # 遍历目录中的所有图像
#         if os.path.isdir(image_dir):
#             for image_name in os.listdir(image_dir):
#                 image_path = os.path.join(image_dir, image_name)
#                 if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     # 根据图片前缀生成标签
#                     image_label = generate_labels_for_image(image_name)
#                     # 写入文件名和对应的标签
#                     writer.writerow([image_name, image_label])
#                     print(f"Annotated {image_name}: {image_label}")

# 根据图片前缀生成标签
def generate_labels_for_image(image_name):
    prefix = image_name.split('_')[0]
    if prefix in labels_dict:
        return f"{', '.join(labels_dict[prefix])}, {prefix}"
    return f"Unknown Category, {prefix}"

def batch_annotate_and_save(image_dir, output_dir):
    # 确保目标文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历目录中的所有图像
    if os.path.isdir(image_dir):
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 根据图片前缀生成标签
                image_label = generate_labels_for_image(image_name)
                # 生成对应的标签文件路径
                label_file_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
                # 写入标签到文件
                with open(label_file_path, 'w') as label_file:
                    label_file.write(image_label)
                print(f"Annotated {image_name}: {image_label}")
if __name__ == "__main__":
    # 示例用法
    image_dir = "/root/autodl-tmp/all_in_one/Meta-FDMixup-main/support_images/train"
    # output_csv = "output_labels.csv"
    output_dir = "/root/autodl-tmp/all_in_one/Meta-FDMixup-main/support_images/train"
    batch_annotate_and_save(image_dir, output_dir)
