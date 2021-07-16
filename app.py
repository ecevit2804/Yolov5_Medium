from flask import Flask, flash, request, redirect, url_for, render_template
import os
import cv2
from werkzeug.utils import secure_filename
from YOLO_DETECT import wheel_detect ## Fonksiyon haline getirilen detect.py kodu
from models.experimental import attempt_load

## Upload edilecek imgelerin yükleneceği klasör, Yolov5 buradan test imgelerini alacak
UPLOAD_FOLDER = os.path.join('static', 'uploads')  
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
## Yolo v5 modelinden geçen imgeler bu klasör yoluna kaydedilecek.
download_path = 'static/downloads/'  
app.secret_key = "secret key"
## Kabul edilecek imge türleri.
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif']) 


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
### Yolo v5 fonksiyonunu çağırdımız fonksiyon.
def process_image(filename): 
    # Eğitimde kaydedilen modelin yolu
    weights = 'runs/train/exp3/weights/best.pt' 
    device = 'cuda' ## GPU-CPU seçimi
    # modeli, device ile load etme
    model = attempt_load(weights, map_location= device) 
    ## Modele girecek imgenin klasör yolu
    i_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
    ### Model sonrası imgelerin klasör yolu
    project = download_path  
    wheel_detect(save_img=True, image=i_path, project=project, weights=weights, model= model, device=device)



@app.route('/')  ## Web arayüzü sayfasının html'i, !! render_template html yollarını templates isimli klasörde arar !!
def home():
    return render_template("index.html")

@app.route('/', methods = ['POST'])
def upload_image():
## Kullanıcının dosya seçip seçmediğini sorgular
    if 'file' not in request.files: 
        flash('Dosya Parçası Bulunamadı')
        return redirect(request.url)
    file = request.files['file']
    print("---------", request.files)
    if file.filename == '':
        flash('Yükeleme için imge seçilmedi')
        return redirect(request.url)
    ## Kullanıcın dosya seçtiği dosyanın uzantısını 14. satıra göre sorgular
    if file and allowed_file(file.filename): 
        ## secure_filename ile dosya adını kullanıcıya asla güvenme ilkesiyle tekrar elde etmiş oluyoruz.
        filename = secure_filename(file.filename) 
        ## kullanıcının seçtiği dosya, istediğimiz klasöre kaydedilir.
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 
        ## kaydettigimiz dosyayı artık Yolo v5 algoritmasında test edebiliriz.
        process_image(filename) 
        
        flash('İmge Başarıyla Yüklendi')
        
        ## upload edilen ve yolodan geçen imgelerin klasör yolu html'e aktarılır
        return render_template('display.html', orjinal_image=os.path.join(app.config['UPLOAD_FOLDER'], filename),
                               process_image = download_path + 'Experiment/' + filename) 
    # Post request yerine, get request gelirse anasayfa gösterilir.
    else:
        flash('Kabul edilen imge türleri: png, jpg, jpeg, gif')
        return redirect(request.url) #



if __name__ == '__main__':
    app.run(debug=True)