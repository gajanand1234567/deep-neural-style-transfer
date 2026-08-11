import os
import time
import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms
import traceback

# Import AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class UploadForm(FlaskForm):
    content      = FileField('Content Image')
    style        = FileField('Style Image')
    content_path = HiddenField()
    style_path   = HiddenField()
    alpha        = FloatField('Alpha', default=1.0)
    submit       = SubmitField('Transfer Style')


# ── Model loading ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

_BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
_VGG_PATH     = os.path.join(_BASE_DIR, 'vgg_normalised.pth')
_DECODER_PATH = os.path.join(_BASE_DIR, 'experiment', 'final_exp', 'decoder_final.pth')

encoder = VGGEncoder(_VGG_PATH).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(_DECODER_PATH, map_location=device))
encoder.eval()
decoder.eval()
print("[INFO] Models loaded successfully.")


# ── Helpers ────────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def style_transfer(content_pil, style_pil, alpha):
    """
    AdaIN style transfer.
    NOTE: vgg_normalised.pth has normalization baked into its first Conv(3->3)
    layer, so we only apply ToTensor() (range [0,1]) — NO external mean/std.
    The training pipeline (get_transform in utils.py) also applies no normalization.
    """
    # 512 gives much better quality — network is fully convolutional, any size works
    tfm = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),       # [0, 1], normalization handled inside VGG
    ])

    c_tensor = tfm(content_pil).unsqueeze(0).to(device)
    s_tensor = tfm(style_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        # encoder() without is_test returns (h1, h2, h3, h4)
        c_feats = encoder(c_tensor)
        s_feats = encoder(s_tensor)

        c4 = c_feats[-1]
        s4 = s_feats[-1]

        print(f"[DEBUG] c4 mean={c4.mean():.4f}  s4 mean={s4.mean():.4f}")

        t      = adaptive_instance_normalization(c4, s4)
        t      = alpha * t + (1 - alpha) * c4
        output = decoder(t)

        print(f"[DEBUG] output range: [{output.min():.3f}, {output.max():.3f}]")

    return output


def tensor_to_pil(tensor):
    """Convert a [1,C,H,W] tensor in [0,1] range to a PIL image."""
    img = tensor.squeeze(0).cpu().clone()
    img = img.clamp(0, 1)           # no denorm — VGG normalises internally
    return transforms.ToPILImage()(img)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    form             = UploadForm()
    result_image     = None
    content_filename = None
    style_filename   = None
    error            = None

    if request.method == 'POST' and form.validate_on_submit():

        # ── Save / retrieve content image ──
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(
                    os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
            else:
                error = 'Content file type not allowed.'
        else:
            content_filename = form.content_path.data or None

        # ── Save / retrieve style image ──
        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(
                    os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
            else:
                error = 'Style file type not allowed.'
        else:
            style_filename = form.style_path.data or None

        # ── Run style transfer ──
        if content_filename and style_filename and not error:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path   = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            try:
                content_pil   = Image.open(content_path).convert('RGB')
                style_pil     = Image.open(style_path).convert('RGB')
                alpha         = float(form.alpha.data)

                output_tensor = style_transfer(content_pil, style_pil, alpha)
                output_pil    = tensor_to_pil(output_tensor)

                # Unique filename prevents any browser caching
                result_filename = f'result_{int(time.time())}_{content_filename}'
                result_path     = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                output_pil.save(result_path)
                print(f"[INFO] Saved result -> {result_path}")

                result_image = result_filename

            except Exception as e:
                traceback.print_exc()
                error = f"Style transfer failed: {e}"

        elif not error:
            error = 'Please upload both a content and a style image.'

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
    )


@app.route('/uploads/<filename>')
def send_image(filename):
    """Serve uploaded / result images for inline display."""
    return send_from_directory(
        os.path.abspath(app.config['UPLOAD_FOLDER']), filename)


@app.route('/download/<filename>')
def download_image(filename):
    """Force-download a result image."""
    return send_from_directory(
        os.path.abspath(app.config['UPLOAD_FOLDER']),
        filename,
        as_attachment=True)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)
