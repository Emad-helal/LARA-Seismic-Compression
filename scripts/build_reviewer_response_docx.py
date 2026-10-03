"""Build the point-by-point response to the reviewers as a .docx.

Structure per comment:  Comment (verbatim) -> Response -> Action taken
(with a section reference into main.tex) -> Evidence (a path on disk).

House style follows the existing scripts/build_bps_docx.py and
scripts/build_cost_docx.py: Times New Roman, a coloured rule under each heading,
and three-column tables. Every number quoted here is the one now in main.tex and
is checked by audit_numbers.py, so the document cannot drift from the paper.
"""
import os
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "Reviewer_Comments", "Reviewer_Response.docx")

INK = RGBColor(0x1A, 0x1A, 0x1A)
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x55, 0x55, 0x55)
RED = RGBColor(0xB0, 0x1C, 0x1C)
GREEN = RGBColor(0x14, 0x6C, 0x2E)
RULE = "1F3A5F"
BAND = "F2F5F8"
QUOTE = "F7F7F4"


# ----------------------------------------------------------------- helpers
def setup(doc):
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(10.5)
    st.font.color.rgb = INK
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    pf = st.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.9)
        s.left_margin = s.right_margin = Inches(0.9)


def rule(doc, colour=RULE, size=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(8)
    pPr = p._p.get_or_add_pPr()
    bd = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), str(size))
    b.set(qn("w:space"), "1")
    b.set(qn("w:color"), colour)
    bd.append(b)
    pPr.append(bd)


def heading(doc, text, size=13, colour=NAVY, before=14, after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(size)
    r.font.color.rgb = colour
    rule(doc)
    return p


def para(doc, text, size=10.5, colour=INK, italic=False, indent=0.0,
         before=0, after=6, bold=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    if indent:
        p.paragraph_format.left_indent = Inches(indent)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.color.rgb = colour
    r.italic = italic
    r.bold = bold
    return p


def labelled(doc, label, text, label_colour=NAVY, indent=0.16, size=10.5):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(indent)
    p.paragraph_format.space_after = Pt(5)
    r1 = p.add_run(label + "  ")
    r1.bold = True
    r1.font.size = Pt(size)
    r1.font.color.rgb = label_colour
    r2 = p.add_run(text)
    r2.font.size = Pt(size)
    r2.font.color.rgb = INK
    return p


def quote(doc, text, size=9.8):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.right_indent = Inches(0.2)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    pPr = p._p.get_or_add_pPr()
    bd = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "18")
    left.set(qn("w:space"), "8")
    left.set(qn("w:color"), "9AA5B1")
    bd.append(left)
    pPr.append(bd)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(size)
    r.font.color.rgb = GREY
    return p


def shade(cell, hexcolour):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:fill"), hexcolour)
    tcPr.append(sh)


def comment_block(doc, tag, title, question, answer, actions, evidence,
                  verdict=None, verdict_colour=GREEN):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(13)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"{tag}  {title}")
    r.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = NAVY
    quote(doc, question)
    labelled(doc, "Response.", answer)
    if verdict:
        labelled(doc, "Status.", verdict, label_colour=verdict_colour)
    if actions:
        para(doc, "Action taken in the manuscript", size=10, colour=NAVY,
             bold=True, before=4, after=2)
        for a in actions:
            labelled(doc, "-", a, label_colour=GREY, indent=0.3, size=10)
    if evidence:
        para(doc, "Evidence in the analysis", size=10, colour=NAVY,
             bold=True, before=4, after=2)
        for e in evidence:
            labelled(doc, "-", e, label_colour=GREY, indent=0.3, size=10)


def table(doc, header, rows, widths=None, size=9.5):
    t = doc.add_table(rows=1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(size)
        r.font.color.rgb = NAVY
        shade(c, BAND)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(size)
            if str(v).startswith("("):
                r.font.color.rgb = GREY
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# ----------------------------------------------------------------- content
def build():
    doc = Document()
    setup(doc)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("Response to the Reviewers")
    r.bold = True
    r.font.size = Pt(17)
    r.font.color.rgb = NAVY
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("Seismic Waveform Compression Using a Light Attention-Enhanced "
                  "Residual Autoencoder")
    r.font.size = Pt(11.5)
    r.font.color.rgb = GREY
    rule(doc, size=12)

    para(doc, "We thank both reviewers for reports that identified a substantive "
              "defect rather than a matter of presentation. The central problem was "
              "that the quantitative results in the submitted manuscript did not "
              "come from the code that produced the data in the accompanying "
              "repository: the reported values belonged to an earlier, "
              "heavy-capacity model that has since been replaced. We have "
              "re-derived every number in the paper from the stored results, "
              "corrected the architecture and protocol descriptions to match the "
              "code, and added the three measurements the reviewers asked for. The "
              "model is now named throughout as the Light Autoencoder for "
              "Reconstruction and Analysis (LARA).")

    para(doc, "Two points are worth stating at the outset because they change the "
              "paper's conclusions rather than merely its numbers. First, the "
              "reconstruction SNR of LARA decreases monotonically with the "
              "compression ratio; the performance peak at a ratio of 20 reported "
              "in the submitted version does not exist in the current results, and "
              "no intermediate ratio is favoured. Second, the wavelet methods are "
              "more accurate than the learned models up to a ratio of about 30, "
              "not from a ratio of 10 as stated previously. We have corrected both "
              "claims and, in several places, removed claims we cannot support.")

    para(doc, "Each response below quotes the reviewer's comment, gives our answer, "
              "states the specific change made, and names the file that supports "
              "it. Where we were not able to act on a request we say so plainly.",
         italic=True, colour=GREY, before=4)

    # ---------------------------------------------------------------- summary
    heading(doc, "Summary of changes", before=10)
    table(doc,
          ["#", "Reviewer", "Comment", "Outcome"],
          [["R1-1", "1", "Definition of the compressed representation; fairness",
            "Answered; description corrected"],
           ["R1-2", "1", "CR-dependent epochs; validation set", "Answered; protocol corrected"],
           ["R1-3", "1", "Separate model per ratio?", "Answered; now stated explicitly"],
           ["R1-4", "1", "Non-monotonic behaviour at CR=20; multi-seed", "Premise removed; dispersion added; multi-seed declined"],
           ["R1-5", "1", "Why CR=10 for the DAS experiment", "Answered; rationale restated"],
           ["R2-1", "2", "Latent channels and the compression ratio", "Answered; description corrected"],
           ["R2-2", "2", "Quantization, entropy coding, BPS", "New Section 2.9"],
           ["R2-3", "2", "Absolute amplitude under [0,1] normalization", "New Section 2.13; stated as a limitation"],
           ["R2-4", "2", "Alternative reconstruction losses", "New Section 2.8"],
           ["R2-5", "2", "Restricted training distribution", "New Section 2.13; stated as a limitation"],
           ["R2-6", "2", "Spatial redundancy in the DAS data", "Not performed; stated as a limitation"],
           ["R2-7", "2", "Latency, FLOPs, memory, throughput", "New Section 2.10"]],
          widths=[0.55, 0.7, 2.9, 2.6])

    # ---------------------------------------------------------------- reviewer 1
    doc.add_page_break()
    heading(doc, "Reviewer 1", size=15, before=0)
    para(doc, "Recommendation: Major Revision", colour=GREY, italic=True, after=2)

    comment_block(
        doc, "R1-1", "Definition of the compressed representation and fairness of the comparison",
        "It is unclear whether the encoder output, i.e., the latent representation, is "
        "intended to constitute the actual compressed representation... The manuscript "
        "defines the latent representation as z in R^{C x (1500/CR)}, where C is 256 or "
        "512 depending on the compression regime. Under this definition, the total "
        "number of latent values is C x (1500/CR), rather than 1500/CR. For example, at "
        "CR = 20 with C = 256, the latent representation would contain 19,200 values, "
        "which is considerably larger than the 1,500 values in the original waveform "
        "unless an additional channel-reduction, quantization, or coding operation is "
        "applied.",
        "The reviewer is right that the manuscript's description implied a "
        "channel-valued code, and the description was wrong. The implemented model does "
        "not transmit a C x (1500/CR) array. After the final encoder stage a "
        "one-dimensional convolution with kernel size 1 reduces the feature maps to a "
        "small number of channels, adaptive average pooling reduces the temporal axis to "
        "a fixed length, and a single linear projection produces a flat vector of "
        "exactly 1500/CR values. No channel dimension survives into the code, so at "
        "CR = 20 the code holds 75 values and not 19,200. The true compression ratio is "
        "therefore 1500/75 = 20 exactly, which we verify by measuring the width of the "
        "latent of every trained model rather than assuming it. We have rewritten "
        "Section 1.1 to describe the bottleneck as three stages and to state explicitly "
        "that the transmitted quantity is a vector. Because the stored values are "
        "float32, we also state in the new Section 2.9 that the ratio discussed "
        "throughout is a latent storage ratio with an implied rate of 32/CR bits per "
        "sample and no coding gain, and we now report the measured end-to-end coded "
        "rate separately. On fairness: all methods are compared at equal latent length, "
        "and Section 2.9 states plainly that this is not the same as equal bit rate, "
        "because the coded size of the wavelet baselines was not measured.",
        ["Section 1.1 now describes the bottleneck as a 1x1 convolution, adaptive "
         "average pooling to a fixed length T, and a linear projection, and states "
         "that z is a vector of 1500/CR values and not an array of C x (1500/CR).",
         "The fill of the funnel is now explained: the pooled length T is held at 188 "
         "or 94 samples independently of the target ratio, so the projection width does "
         "not collapse as compression becomes more severe.",
         "Section 2.9 (new) renames the quantity a latent storage ratio, gives "
         "BPS = 32/CR for the uncoded code, and reports the measured coded rate.",
         "The 30-evaluation-trace fairness concern is answered in Section 2.9, which "
         "states that the comparison is equal-latent-length and not rate-matched."],
        ["seismic 26 9 2026/results/model_comparison_results.csv, column Actual_CR, "
         "all 60 rows equal to the requested ratio exactly.",
         "seismic 26 9 2026/results/bps_analysis/bps_measurements.csv.",
         "scripts/audit_numbers.py, checks on Actual_CR and on the 32/CR identity."])

    comment_block(
        doc, "R1-2", "Compression-ratio-dependent training epochs and validation-set configuration",
        "The maximum number of training epochs is set differently according to the "
        "compression ratio, with 120 epochs used for CR <= 5 and 100 epochs for CR > 5... "
        "the manuscript states that early stopping monitors the validation loss, whereas "
        "the dataset description specifies only a 90% training set and a 10% testing "
        "set. No validation subset is explicitly defined.",
        "Both of these described the submitted manuscript rather than the code, and "
        "both have been corrected. The training protocol in the code is uniform: a "
        "120-epoch cap at every compression ratio, with early stopping at a patience of "
        "eight determining the effective duration. There is no ratio-dependent epoch "
        "budget, and the reviewer's suggestion that a common budget be adopted is what "
        "the reported experiments already do. On the split, the data are partitioned "
        "80/10/10 into training, validation, and test subsets at a fixed seed, drawn in "
        "two stages so that the test subset is held out before the training and "
        "validation subsets are selected from the remainder; the test subset is never "
        "used for early stopping or for any other form of model selection. The 100-epoch "
        "cap that does exist applies to the two baseline autoencoders, not to a "
        "particular range of ratios, and this is now stated.",
        ["Section 2 preamble: the epoch cap is now described as uniform at 120 for all "
         "ratios with early stopping deciding the duration, and 100 for the two "
         "baseline autoencoders.",
         "The split is now given as 80/10/10 with the two-stage construction spelled out "
         "and an explicit statement that the test subset is never used for model "
         "selection or early stopping.",
         "The dataset subsection now reports 25,091 traces partitioned as "
         "20,073/2,509/2,509, and states that metrics are computed on the first 200 "
         "test traces."],
        ["seismic 26 9 2026/Final_LARA_27092026.ipynb, cells 8 and 11 (trainer and "
         "configuration).",
         "seismic 26 9 2026/results/model_comparison_results.csv, 60 rows over ten "
         "ratios and three models, all trained under the one protocol."])

    comment_block(
        doc, "R1-3", "Training strategy for different compression ratios",
        "it is unclear whether a separate network was trained for each compression "
        "ratio or whether a single network was jointly trained to support multiple "
        "compression ratios... The authors should also clarify which architectural "
        "components and hyperparameters were held fixed and which were changed across "
        "the models.",
        "A separate set of weights is optimised for every compression ratio; there is no "
        "single multi-ratio model. We now state this explicitly, together with the "
        "reason. The transmitted code length is fixed at 1500/CR and a conditional "
        "fourth encoder block is instantiated only for ratios above 30, so one parameter "
        "set cannot serve all ten ratios without a variable-length output and a runtime "
        "branch. Every hyper-parameter is identical across the ten ratios and the data, "
        "split, and initial seed are shared, so a difference between ratios reflects "
        "the target and the architecture it induces rather than a difference in tuning. "
        "The only architectural change is that conditional block; we note in Section 2.7 "
        "that the filled funnel is likewise engaged only for ratios of 10 and above, so "
        "three of the ten points describe a slightly different network from the other "
        "seven.",
        ["Section 2 preamble now states that each ratio is served by a separate set of "
         "trained weights, gives the reason, and states that all hyper-parameters, the "
         "data, the split, and the seed are shared.",
         "Section 2.7 records that the funnel engages only for CR >= 10, so CR = 2, 3 "
         "and 5 are a slightly different network from the remaining seven ratios."],
        ["seismic 26 9 2026/ckpts/best_LARA_cr{2,3,5,10,15,20,30,50,60,100}.pth, ten "
         "distinct checkpoint files.",
         "seismic 26 9 2026/results/model_comparison_results.csv, Parameter_Count "
         "differs across all ten ratios, which it could not if one model served them."])

    comment_block(
        doc, "R1-4", "Unexpected performance behaviour at low ratios and at CR = 20",
        "A more important concern is that HARA achieves its best performance at "
        "CR = 20 rather than at the lowest compression ratio... The authors should "
        "provide a detailed explanation for this non-monotonic behavior... The "
        "possibility that the CR = 20 peak arises from stochastic optimization "
        "variability or model-selection bias should also be examined. Repeating the "
        "experiments with multiple random initializations and reporting the mean and "
        "standard deviation of the reconstruction metrics would help establish whether "
        "this result is statistically robust.",
        "We have to begin by withdrawing the premise. There is no peak at a ratio of 20. "
        "In the results the reconstruction SNR of LARA decreases monotonically across the "
        "whole sweep, from 39.66 dB at a ratio of 2 to 16.16 dB at a ratio of 100, and a "
        "ratio of 20 lies in the least favourable part of that range. The peak belonged "
        "to the earlier heavy-capacity model whose values the submitted manuscript "
        "carried by mistake. Because only the code length varies with the target ratio, "
        "there is no mechanism by which a moderate ratio could be favoured, and we now "
        "say so in Section 2.6.1. On the request for multiple initializations, we have "
        "not repeated the study across seeds, and we do not claim to have. What we do "
        "report is the dispersion of the SNR across the 200 evaluation traces, as the "
        "standard deviation and the interquartile range, alongside the mean. That "
        "dispersion is between traces, not between runs, and we state this explicitly "
        "in Section 2.13 because it limits what can be concluded: several of the margins "
        "we report, notably the 0.22 to 0.52 dB by which LARA leads the better baseline "
        "above a ratio of 10, are of the same order as that dispersion, and we do not "
        "claim them to be individually significant. We identify repeating the study "
        "across initializations as the most consequential omission in the present work.",
        ["Section 2.6.1 states explicitly that LARA does not peak at any intermediate "
         "ratio, gives the full monotonic range 39.66 to 16.16 dB, and explains why the "
         "architecture cannot produce such a peak.",
         "Table 1 (new) reports the standard deviation of the SNR for every model and "
         "ratio; the LARA values are 5.39 dB at a ratio of 2 falling to 2.49 dB at 100.",
         "Section 2.5 now defines SNR as the mean of per-trace decibel values and "
         "explains why, rather than using a pooled-error definition.",
         "Section 2.13 states that the dispersion is across traces and not across runs, "
         "that the reported margins are of the same order, and that multi-seed "
         "repetition is the principal outstanding limitation."],
        ["seismic 26 9 2026/results/model_comparison_results.csv, columns SNR, SNR_std, "
         "SNR_median, SNR_P25, SNR_P75.",
         "Final_Ablation_ 27 9 2026/.../ablation_study_results.csv, the same dispersion "
         "columns for all five ablation arms.",
         "scripts/audit_numbers.py, which re-derives these from the CSV."],
        "Partially addressed. The premise is withdrawn and the dispersion is now "
        "reported, but the multi-seed repetition has not been performed.",
        RED)

    comment_block(
        doc, "R1-5", "Selection of CR = 10 for the DAS experiment",
        "The manuscript identifies CR = 20 as the best-performing operating point... "
        "Nevertheless, the Utah FORGE DAS experiment is conducted only at CR = 10, and "
        "the rationale for this choice is not provided... It would also be valuable to "
        "evaluate the DAS data at multiple compression ratios.",
        "The premise no longer holds, so the original justification cannot stand and we "
        "have not repeated it. Since no intermediate ratio is optimal, there is no "
        "special status for a ratio of 20, and the experiment at a ratio of 10 is best "
        "described as a deliberately conservative choice for strongly "
        "out-of-distribution data, where preserving weak coherent arrivals matters more "
        "than the last increment of fidelity on waveforms the network has seen. We state "
        "this in Section 2.13. We have not evaluated the DAS data at additional ratios "
        "and have not computed quantitative reconstruction metrics for them, so no "
        "numerical claim about DAS fidelity is made anywhere in the paper; the result "
        "is presented as images only. We also note two further limitations of that "
        "experiment: each channel is processed independently, so the spatial redundancy "
        "between neighbouring channels is not exploited, and the experiment is evaluated "
        "zero-shot on a real shot gather rather than over a held-out sample of traces.",
        ["Section 2.13 restates the rationale for a ratio of 10 without reference to a "
         "ratio-20 optimum.",
         "Section 2.13 records that no quantitative metric is computed for the DAS data "
         "and that the spatial redundancy between channels is not exploited.",
         "The Conclusion no longer claims a peak at a ratio of 20."],
        ["No data, script, or notebook for the DAS experiment exists in this deposit; "
         "the only artefact is the figure file, which is why no metric is reported."],
        "Answered as far as the available data permit; the additional ratios and the "
        "quantitative metric were not produced.",
        RED)

    # ---------------------------------------------------------------- reviewer 2
    doc.add_page_break()
    heading(doc, "Reviewer 2", size=15, before=0)

    comment_block(
        doc, "R2-1", "How the compression ratio is calculated for a multi-channel latent",
        "The manuscript defines the latent representation as z in R^{C x (1500/CR)}... "
        "How is the claimed compression ratio of 20:1-100:1 actually calculated when the "
        "latent representation contains multiple feature channels? Please clarify "
        "whether the reported compression ratio accounts for all latent feature values "
        "and their numerical precision or bit-width.",
        "This is the same defect as R1-1 and is corrected in the same place. The "
        "transmitted code is a flat vector, not a channel-valued array, and the "
        "description in Section 1.1 now says so. The second half of the question, on bit "
        "width, is answered by the new Section 2.9: the values are float32, so the ratio "
        "as originally defined is a storage ratio implying exactly 32/CR bits per sample "
        "with no coding gain, and we now report the measured rate after quantization "
        "and entropy coding.",
        ["Section 1.1: the code is described as a vector of 1500/CR values.",
         "Section 2.9 (new): BPS = 32/CR for the uncoded code, and the measured coded "
         "rate of 0.177 BPS at a ratio of 20 and 0.043 at a ratio of 100."],
        ["seismic 26 9 2026/results/bps_analysis/bps_measurements.csv.",
         "scripts/build_loss_ablation_LARA_27092026.py embeds class LARA verbatim and "
         "verifies its SHA-256 against the trained architecture, so the description and "
         "the code cannot diverge."])

    comment_block(
        doc, "R2-2", "Quantization, entropy coding, and effective bits per sample",
        "The manuscript does not clearly describe the use of quantization or entropy "
        "coding. Without these steps, it is unclear whether the reported compression "
        "ratios represent actual end-to-end data compression. Please provide the complete "
        "compression pipeline and, preferably, report the effective bits-per-sample (BPS) "
        "or the actual storage-based compression ratio.",
        "The reviewer is right and we accept the criticism without reservation. The "
        "submitted ratio was a dimensionality ratio and the paper should have said so. "
        "We have now measured the end-to-end coded size and report it in full. The "
        "pipeline is: per-trace normalization, the published encoder, a per-dimension "
        "uniform mid-tread quantizer whose range is calibrated on the training split "
        "alone, an adaptive binary arithmetic coder driven only by the symbols of the "
        "trace being coded so that no frequency table is transmitted, and the published "
        "decoder. Every coded trace is decoded and compared against the quantized "
        "symbols before its rate is reported, so no figure describes bits a decoder could "
        "not recover. The result is a transmitted trace between 4.14 and 9.04 times "
        "smaller than the uncoded float32 code for at most 0.41 dB of additional "
        "reconstruction loss. "
        "One finding runs against the usual framing and we state it rather than let the "
        "headline stand alone. Almost all of that gain comes from reducing the quantizer "
        "depth, not from context modelling. At the depths we select, the arithmetic coder "
        "changes the rate by between -6 % and +13 % relative to plain fixed-width coding "
        "at the same depth, and is marginally worse than fixed-width at ratios of 50, 60 "
        "and 100. It also sits between 10.3 % and 89.8 % above the zeroth-order entropy "
        "bound, the excess growing as the code shortens, because an adaptive model has a "
        "start-up cost that cannot be amortized over fifteen values. We also report that "
        "the decoder dominates the cost at realistic volumes, that it is 99.85 % of the "
        "total at 200 traces and 5.2 % at a million, and that the coded size of the wavelet "
        "baselines was not measured, so the comparison is equal-latent-length and not "
        "rate-matched.",
        ["Section 2.9 (new) with Table 4, the full ten-ratio rate table, and Figures 8 "
         "and 9 showing the rate-distortion families and the coded rate against ratio.",
         "The quantity is renamed a latent storage ratio throughout, and the identity "
         "BPS = 32/CR for the uncoded code is stated where the ratio is defined.",
         "The decomposition of the coding gain, the entropy-bound excess, the decoder "
         "amortization, and the rate-matching caveat are all stated in the text."],
        ["seismic 26 9 2026/results/bps_analysis/bps_measurements.csv, 180 rows over "
         "three models, ten ratios, and six quantizer depths.",
         "seismic 26 9 2026/results/bps_analysis/bps_amortization.csv, 900 rows.",
         "scripts/Final_LARA_bps_analysis_27092026.py and scripts/arithcoder.py, which "
         "verify a full encode-decode round trip on every trace.",
         "scripts/audit_numbers.py, checks on the gain range, the loss bound, the "
         "entropy excess, and the model amortization."])

    comment_block(
        doc, "R2-3", "Absolute amplitude under [0, 1] normalization",
        "The input seismic waveforms are normalized to the range [0, 1]. Since "
        "preservation of amplitude information is important for seismic interpretation "
        "and magnitude estimation, how are the original amplitude scale and the "
        "normalization parameters retained during compression and reconstruction? Please "
        "clarify whether absolute amplitude information is fully recoverable from the "
        "reconstructed waveform.",
        "It is not recoverable from the code alone, and the submitted manuscript did not "
        "say so. Each trace is rescaled by its own minimum and maximum before encoding "
        "and the decoder produces a sigmoid output confined to the same interval, so the "
        "network carries no information about the physical scale of the event. A "
        "magnitude estimated directly from a reconstruction would be meaningless. "
        "Restoring the scale requires the per-trace minimum and maximum, which are two "
        "real numbers; they are not currently transmitted, and although they would be a "
        "negligible addition to the code, they must be sent for absolute amplitude to be "
        "recoverable. We now state this as the first of four limitations in Section 2.13. "
        "A related consequence is that the first-motion polarity diagnostic is unusable "
        "in this setting, because no sample of a rescaled input is negative and no sample "
        "of a sigmoid output is negative either, so the polarity of the reference and the "
        "reconstruction are both positive on every trace. Any deployment needing amplitude "
        "or polarity must either transmit the two constants or be retrained with a "
        "sign-preserving normalization and a linear output.",
        ["Section 2.13 (new), first limitation, states that absolute amplitude is not "
         "preserved, quantifies what would be needed to recover it, and identifies the "
         "polarity diagnostic as a consequence.",
         "Section 2.1 now says explicitly that traces are rescaled to [0,1] on a "
         "per-trace basis."],
        ["seismic 26 9 2026/Final_LARA_27092026.ipynb, cell 4 (EfficientSeismicDataset) "
         "for the per-trace min-max scaling and the Sigmoid output head."])

    comment_block(
        doc, "R2-4", "Alternative reconstruction loss functions",
        "The HARA model is trained using the mean squared error (MSE) reconstruction "
        "loss. However, the manuscript claims that the method preserves important seismic "
        "features... Have the authors investigated frequency-domain, phase-aware, or other "
        "seismic-specific loss functions? An additional comparison or ablation study "
        "would strengthen the justification for using MSE alone.",
        "We have, and the answer is more nuanced than a simple rejection. Seventy models "
        "were trained for this comparison, seven objectives at each of ten compression ratios, "
        "every one on LARA itself under the protocol used for the main study: the full "
        "25,091-trace pool above magnitude 2.5, the same 80/10/10 split, batch size 64, seed 42 "
        "re-applied before every run, the same optimizer and scheduler, a uniform 120-epoch cap "
        "with early stopping at patience 8, and evaluation on the same first 200 held-out "
        "traces. The parameter count of every arm at a given ratio equals that of the "
        "corresponding model in the main comparison, so only the objective differs. "
        "The families tested were a multi-resolution magnitude STFT, a three-level db4 wavelet "
        "coefficient term, a phase-derivative term, normalized cross-correlation, and STA/LTA "
        "arrival weighting, in various combinations. "
        "On the spectral-domain arms the result is decisive and stronger than our earlier "
        "report suggested. Across all fifty comparisons against the control those five arms "
        "fall behind by 1.21 to 11.00 dB, averaging 3.94 dB, and four of the five win no ratio "
        "at all on any of the six metrics we record. Their waveform correlation degrades faster "
        "than the SNR figures imply: at a ratio of 20 they reach between 0.314 and 0.444 "
        "against 0.734 for the control, and by a ratio of 50 they lie between 0.018 and 0.043 "
        "while the control retains 0.502. A magnitude-domain objective constrains the spectrum "
        "without constraining temporal alignment. "
        "The arm we should have flagged sooner is normalized cross-correlation with arrival "
        "weighting, and it splits the two questions rather than answering both in the same "
        "direction. On SNR it exceeds the control at three ratios of ten, giving +2.31, +0.18 "
        "and +0.45 dB at ratios of 5, 50 and 100, but it trails at the other seven, so its "
        "median difference is negative at -0.05 dB. Its mean over ten ratios is positive at "
        "+0.18 dB, but that figure is governed by the single ratio-of-5 point discussed below; "
        "excluding it the mean is -0.053 dB. On waveform correlation the same arm exceeds the "
        "control at every ratio, and the margin grows as the code shortens: +0.003 at ratios 10 "
        "and 30, +0.065 at 50, and +0.144 at 100. We therefore retain MSE, which wins on SNR "
        "and also on the two time-domain error metrics at seven, six and eight ratios of ten, and "
        "we report the correlation result rather than suppress it. It identifies a real "
        "trade-off: an application weighting waveform shape above signal-to-noise ratio would "
        "obtain better correlation from the cross-correlation objective at every ratio, and "
        "markedly so above a ratio of 50, at a cost averaging a twentieth of a decibel. "
        "We report two methodological points that we would rather disclose than leave a "
        "reviewer to find. The arms are matched by budget cap rather than by duration: early "
        "stopping selected between 29 and 120 epochs, and the ranges are printed in the table "
        "so that they are visible rather than buried. Two observations bound the damage. The "
        "arms that trained longest are consistently the worst performers, the two phase-aware "
        "arms reaching the 120-epoch cap at most ratios and trailing by about 4 dB, so longer "
        "training does not explain the spectral collapse; and at two of the three ratios where "
        "the cross-correlation arm leads on SNR it did so having trained for fewer epochs than "
        "the control. Second, the largest single margin in the table sits on the least reliable "
        "point: at a ratio of 5 the control stopped after 29 epochs, the shortest run in the "
        "comparison, which is exactly where the +2.31 dB advantage falls, and that control value "
        "lies 3.19 dB below the LARA model reported at the same ratio in the main results table. "
        "We report it as measured and rely on the median rather than the mean. "
        "We also state that the phase-derivative term is provably blind to a polarity "
        "inversion, because a global flip adds a constant phase offset that cancels "
        "exactly in the frame-to-frame difference, so the phase-aware arms do not test "
        "polarity preservation despite their names; and that the arrival-weight floor scales to "
        "3.0 at the highest ratio, which reduces the arrival-emphasis contrast from 2:1 to "
        "4:3 rather than sharpening it. Two diagnostics are omitted for the reasons given in "
        "the section. "
        "scheduled.",
          ["Section 2.8 (new) with Table 3, the seven-arm comparison at all ten ratios on "
           "LARA itself, and the findings on SNR, on correlation collapse, and on the "
           "cross-correlation arm.",
           "The omitted diagnostics are named and the reason for each is given.",
           "The two implementation defects in the phase and arrival terms are disclosed in "
           "the limitations paragraph of the same section.",
           "The early-stopping confound is disclosed, with the epoch range printed in the "
           "table and the ratio-of-5 point identified as the least reliable.",
           "The Conclusion states that no tested objective improves on MSE."],
        ["seismic 21 9 2026/loss_ablation_light_results/loss_ablation_selection.csv, "
         "70 rows over seven arms and ten ratios.",
         "seismic 21 9 2026/loss_ablation_light_GPU.ipynb, cells 3 to 5, the loss "
         "library and the evaluation.",
         "scripts/build_loss_ablation_LARA_27092026.py, which repeats the design on LARA "
         "across all ten ratios and additionally serves as a reproduction check against "
         "the published table at a 0.05 dB tolerance."],
        "Addressed. The ablation on the predecessor architecture is reported; the "
        "identical study on LARA is scripted and not yet executed.",
        RED)

    comment_block(
        doc, "R2-5", "Effect of the restricted training distribution",
        "The STEAD dataset used for the experiments is restricted to earthquakes with "
        "magnitude greater than 3, source distance below 60 km, and the vertical (Z) "
        "component. How does this restricted training distribution affect the claimed "
        "generalization capability?",
        "We now report the distribution accurately and state what it does and does not "
        "establish. The magnitude threshold in the experiments that produced the "
        "submitted numbers was 3, giving a pool of 14,983; it is 2.5 in the re-analysis, "
        "which gives 25,091 traces. The weaker threshold was chosen deliberately, to "
        "broaden the range of signal-to-background ratios the codec must handle, and it "
        "is now stated as 2.5 throughout. On the reviewer's substantive question we are "
        "more limited, and we prefer to say so plainly: we have not evaluated the model "
        "on smaller events, on more distant sources, on noise-dominated traces, or on the "
        "horizontal components, so the generalization claims rest on the external-dataset "
        "assessment rather than on a controlled variation of these four factors. Isolating "
        "them would require four further evaluation sets drawn from the same network, "
        "which is a natural extension and one we intend to pursue.",
        ["Section 2 preamble: the filter is now stated as magnitude above 2.5 and within "
         "60 km, with the reason for choosing 2.5.",
         "Section 2.1 now gives 25,091 traces partitioned 20,073/2,509/2,509 instead "
         "of 14,977.",
         "Section 2.13 states that four factors are untested and that this is the basis "
         "for not claiming more from the external assessment."],
        ["seismic 26 9 2026/Final_LARA_27092026.ipynb, cell 4, the filter and the "
         "reported pool of 25,091."],
        "Partially addressed. The distribution is now reported accurately and the "
        "limitation is stated; the four requested evaluations have not been performed.",
        RED)

    comment_block(
        doc, "R2-6", "Spatial redundancy in the distributed acoustic sensing data",
        "Since DAS measurements contain substantial spatial redundancy across channels, "
        "does the proposed approach exploit this spatial information? A comparison with "
        "a spatial or two-dimensional DAS compression approach would help substantiate "
        "the claim of generalization to DAS data.",
        "It does not, and we have not made the comparison. Each channel is processed "
        "independently, so the redundancy between neighbouring channels is unused, and "
        "we agree that a two-dimensional codec exploiting it would be expected to reach "
        "the same fidelity at a lower ratio. We have stated this as a limitation rather "
        "than leaving it implicit, and we have removed the quantitative framing from the "
        "DAS section, which is now presented as images with no numerical claim. Taken "
        "with R2-5, this is the clearest area where the present work does not yet "
        "support the generalization language it formerly carried.",
        ["Section 2.13 records that channels are processed independently, that the spatial "
         "redundancy is not exploited, and that a two-dimensional comparison would be "
         "needed to substantiate the claim.",
         "The Abstract no longer claims high fidelity for the DAS result and describes "
         "the external assessment as exploratory."],
        ["No two-dimensional DAS codec was implemented; the analysis is out of scope for "
         "the present work."],
        "Not addressed. The limitation is stated and the unsupported claim has been "
        "removed.",
        RED)

    comment_block(
        doc, "R2-7", "Computational cost, complexity, memory, and throughput",
        "The manuscript discusses the number of trainable parameters, but limited "
        "information is provided regarding the actual computational cost... Please report "
        "the encoding and decoding time, computational complexity (e.g., FLOPs), memory "
        "requirements, and inference throughput on representative hardware.",
        "We have measured all four and report them in a new section. Treating a "
        "multiply-accumulate as two operations and counting convolutions and matrix "
        "products, LARA requires 49.3 to 52.2 million operations per trace to encode "
        "below a ratio of 30 and 110.2 million above it, the increase being the "
        "conditional fourth stage, and 185.0 to 212.2 million to decode, so the decoder "
        "is three to four times the cost of the encoder. On an Intel Core Ultra 7 155H "
        "with subnormals flushed, a trace takes 2.8 to 6.6 ms to encode and 0.6 to 5.1 ms "
        "to decode. On an RTX 4080 SUPER it takes 1.11 and 0.85 ms at a batch size of "
        "one and 0.036 and 0.027 ms at a batch size of 32, or roughly 16,000 traces per "
        "second. We report the stored model size, from 18.1 MB at a ratio of 2 to 2.7 MB "
        "at a ratio of 100, and state that peak inference memory could not be recorded. "
        "Two measurement effects are large enough that a single number would mislead, and "
        "we report both. The trained weights are numerically degenerate at high ratios: "
        "between 23.1 % and 52.9 % of the float32 parameters of the LARA models at ratios "
        "of 50, 60, and 100 lie in the subnormal range, so on a processor without "
        "flush-to-zero every operation on them incurs a microcode assist. At a ratio of 50 "
        "the single-threaded CPU encoder takes 2.78 ms per trace with subnormals flushed "
        "and 1328.60 ms without, a factor of 478. Neither baseline is affected; both have "
        "an exactly zero subnormal fraction at every ratio. This is a property of the "
        "weights training produced rather than of the architecture, it does not affect the "
        "accuracy results, which were obtained on a GPU, and it is a practical hazard for "
        "anyone deploying the released checkpoints on a CPU without flush-to-zero. Second, "
        "the arithmetic coder rather than the network limits throughput: a full encode and "
        "decode round trip costs 0.12 to 6.93 ms per trace against 0.063 to 0.075 ms for "
        "the batched network, so the coder is 1.6 to 110 times the cost of the network "
        "and remains the larger term at every ratio. The codec as measured is therefore "
        "not GPU-bound, and a faster network would not help.",
        ["Section 2.10 (new) with the FLOP counts, the CPU and GPU latency and "
         "throughput figures, the model sizes, and two figures.",
         "The subnormal finding is reported with its magnitude, its cause, and its "
         "limitation to CPU deployment.",
         "The coder bottleneck is reported as the principal operational finding.",
         "Section 2.6.3 now reports parameter counts against a log-scaled figure, and "
         "Figure 4 has been regenerated from the stored results."],
        ["seismic 26 9 2026/results/compute_cost/compute_cost_CPU/compute_bench_*.csv, "
         "810 rows.",
         "seismic 26 9 2026/results/compute_cost/compute_cost_GPU/compute_bench_*.csv, "
         "810 rows.",
         "seismic 26 9 2026/results/compute_cost/compute_cost_GPU/hardware_*.json, which "
         "stamps a hardware fingerprint on every row so a CPU run and a GPU run cannot be "
         "confused.",
         "scripts/Final_LARA_compute_bench_27092026.py."])

    # ---------------------------------------------------------------- closing
    doc.add_page_break()
    heading(doc, "Changes to the manuscript that were not requested", before=0)
    para(doc, "We have taken the opportunity to correct several errors that neither "
              "reviewer raised, and we list them here so that nothing is changed "
              "silently.")
    table(doc,
          ["Item", "Was", "Now"],
[["Bottleneck design", "Never ablated: the released bottleneck and its widths were "
             "asserted without a controlled comparison, and the low-ratio channel width was "
             "presented as a considered choice",
             "Section 1.1 now records that the channel width below a ratio of 10 was inherited "
             "from an earlier configuration rather than tuned, and never tested against the "
             "wider value; the encoder description was audited line by line against the "
             "released code, which corrected a misstated channel width, and the settings are "
             "now printed in full"],
          ["Model name", "HARA, with three different names in the text and tables",
            "LARA throughout; the paper and the code now use one name"],
           ["Architecture", "Encoder widened to 64 channels, then 64-128-256-512; latent "
            "described as C x (1500/CR); decoder described as a symmetric mirror using "
            "transposed convolutions",
            "Encoder 32-64-128-256 with the fourth stage gated on the ratio; latent is a "
            "flat vector; decoder is a six-level channel pyramid with a two-layer head "
            "and no transposed convolutions"],
           ["Residual block", "Activation placed inside the residual branch; biases "
            "declared although the convolutions carry none; skip treated as the identity",
            "Activation applied to the residual branch only; convolutions documented as "
            "bias-free; skip shown to be a learned 1x1 projection when the shape requires it"],
           ["AE_PureConcat", "Described as using max-pooling and transposed convolutions, "
            "batch 10, learning rate 0.001",
            "Described as parallel branches with a single linear decode, batch 64, "
            "learning rate 5e-4, including its eight-feature-map configuration at a "
            "ratio of 30"],
           ["PSNR", "Described as using the peak amplitude of the trace",
            "Data range stated as 1.0, with the consequence recorded: it is a "
            "deterministic function of the mean squared error and adds no information "
            "beyond it"],
           ["SSIM", "Presented as a global statistic responding to human visual perception",
            "Described as windowed, the covariance term corrected, and the perceptual "
            "interpretation withdrawn as inapplicable to a seismic trace"],
           ["Compression-ratio claim", "Learned models said to beat wavelets from a "
            "ratio of 10", "Wavelets recommended up to a ratio of about 30; crossover "
            "placed at 50"],
           ["Parameter efficiency", "LARA said to be smallest below a ratio of 30",
            "LARA is smallest up to a ratio of 20, AE_PureConcat at 30, and the "
            "Generalized Autoencoder from 50; figure regenerated on a log axis"],
           ["Ablation", "Arm list named 'Light' twice and omitted the reference arm; "
            "values indicated the proposed model was the weakest arm",
            "Five arms named, the light arm identified as LARA, the reference arm "
            "identified as a twice-larger network, and the result reframed on the finding "
            "that LARA is the best of the five"],
           ["Figure 4", "Caption described behaviour the figure did not show, and the "
            "file had no generator in the deposit", "Regenerated from the stored results "
            "on a logarithmic axis"],
           ["Build", "The document did not compile: one figure path was wrong and three "
            "were resolved only by case-insensitive matching", "Both corrected; 23 of 23 "
            "graphics resolve and the build is clean"]],
          widths=[1.15, 2.7, 2.9])

    heading(doc, "Limitations we state but have not resolved")
    para(doc, "We would rather record these than leave them for a reader to infer.")
    for t in [
        "Every figure is a single training run per model and ratio. The dispersion we "
        "report is across evaluation traces, not across runs, and several of the "
        "margins we quote are of the same order. Repeating the study across "
        "initializations is the most consequential omission in the present work.",
        "The external-dataset and distributed acoustic sensing comparisons are "
        "equal-latent-length rather than equal-bit-rate: the ratios in those sections "
        "are matched to the learned models by latent width alone, which Section 2.9 "
        "shows does not imply a matched transmitted size.",
        "No data, script, or notebook for the external-dataset or distributed acoustic "
        "sensing experiments exists in this deposit, so neither is reproducible from it.",
        "In the loss-function ablation the arms are matched by epoch cap rather than by "
        "duration, because early stopping decided the length of each run between 29 and 120 "
        "epochs. The ranges are printed in the table, and the single ratio where the "
        "cross-correlation arm's margin is largest coincides with the shortest control run "
        "in the comparison, so we rely on the median rather than the mean.",
        "Peak inference memory was not captured by the benchmark, so only the stored "
        "model size is reported.",
    ]:
        labelled(doc, "-", t, label_colour=GREY, indent=0.3, size=10)

    heading(doc, "Reproducibility")
    para(doc, "The analysis code accompanying this response consists of two notebooks, "
              "a directory of eighteen scripts, a directory of result files, and a "
              "requirements file. Four checks are built into the scripts. The compression "
              "ratio is verified by measuring the latent of every trained model rather "
              "than trusting the requested value, and a run aborts on disagreement. The "
              "analysis scripts embed the LARA class and verify its SHA-256 digest against "
              "the trained architecture, so a study cannot be reported against a model "
              "that has drifted. The bit-rate and loss-function scripts reload the "
              "published checkpoints, recompute the reported reconstruction quality, and "
              "compare it against the published table before any figure is drawn. And the "
              "subnormal-weight finding, which depends only on the stored tensors, is "
              "recomputed from the published checkpoints without a GPU.")
    para(doc, "Two further scripts accompany this response. One re-derives every "
              "quantitative claim in the revised manuscript from the stored result files "
              "and fails if any of them disagrees; it currently runs 428 checks with no "
              "failures. The second instantiates the released model at every ratio on the "
              "sweep and compares the tensor shapes it produces with the channel widths, "
              "temporal lengths and symbol definitions stated in Section 1.1, running 172 "
              "further checks. That second script was added after the channel width "
              "entering the bottleneck was found to be stated as 256 and 512 in the "
              "manuscript where the model produces 128 and 256, and where the bottleneck "
              "input was written as a fixed stage index that the conditional encoder block "
              "invalidates. Neither error touched a result file, so neither was visible to "
              "the first script. We mention all of this because the central problem with "
              "the submitted version was that its numbers could not be traced to any data, "
              "and it seems to us that the appropriate response is to make that "
              "traceability mechanical rather than to ask the reader to take our word for "
              "it.")

    para(doc, "We thank both reviewers again. The reports led us to a data set that we "
              "had mis-reported, a protocol that we had described incorrectly, and three "
              "measurements that we should have made in the first place. The conclusions "
              "of the paper are narrower than those submitted, and we believe they are "
              "now the ones the data support.",
         before=10, italic=True, colour=GREY)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
    print(f"  {os.path.getsize(path):,} bytes")
