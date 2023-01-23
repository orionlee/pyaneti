from IPython.display import display, Image, HTML
from pathlib import Path
import lightkurve_ext_pyaneti as lkep

def char_list_inclusive(c1, c2):
    """Return the list of characters from `c1` to `c2`, inclusive."""
    return [chr(c) for c in range(ord(c1), ord(c2) + 1)]

def show_image(img_path):
    if img_path.exists():
        return display(Image(img_path))
    else:
        return display(HTML(f"[ Image <code>{img_path}</code> not found. ]"))

def display_results (TIC_no, target_out_dir):
    show_image(Path(target_out_dir, f"{TIC_no}_posterior.png"))
    #show_image(Path(target_out_dir, f"{alias}_correlations.png"))
    show_image(Path(target_out_dir, f"{TIC_no}_lightcurve.png"))
    show_image(Path(target_out_dir, f"{TIC_no}b_tr.png"))
    
    # read processed LC data from Pyaneti output 

    #filename = Path(target_out_dir, f"{TIC_no}-trdata_lightcurve.txt")
    #lc_tr = lkep.read_pyaneti_lc_dat(filename, time_format="btjd")

    filename = Path(target_out_dir, f"{TIC_no}-trmodel_lightcurve.txt")
    lc_model = lkep.read_pyaneti_lc_dat(filename, time_format="btjd")

    #ax = lc_tr.scatter(label=f"TIC {TIC_no} processed by Pyaneti");
    #ax = lc_model.plot(ax=ax, label="model");

    # read full data file
    
    filename = Path(target_out_dir, f"full_{TIC_no}_data.txt")
    lc_full = lkep.read_pyaneti_lc_dat(filename, time_format="btjd")
    
    bin_time = 30/24/60
    lc_binned = lc_full.bin(bin_time)
    
    ax = lc_full.scatter(label=f"TIC {TIC_no}",color="lightblue");
    ax = lc_binned.plot(ax=ax,label="Binned",color="dodgerblue")
    ax = lc_model.plot(ax=ax, label="Pyaneti model", linewidth = 1.5);
    
    ax1 = lc_full.scatter(label=f"TIC {TIC_no}",color="paleturquoise");
    ax1 = lc_binned.plot(ax=ax1,label="Binned",color="green", linewidth=0, markersize=2, marker=".")
    ax1 = lc_model.plot(ax=ax1, label="Pyaneti model", linewidth = 1.5);
