import streamlit as st
import pytesseract
import easyocr
import torch
from PIL import Image
import fitz  # PyMuPDF
import json
import time
import os
from pathlib import Path
import numpy as np
import cv2
from io import BytesIO
import base64

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass
except Exception as e:
    # .env file has issues, print warning but continue
    print(f"Warning: Could not load .env file: {e}")

# Try to import doctr (might not be available)
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False
    # Don't show warning here, we'll handle it in the UI

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    # Don't show warning here, we'll handle it in the UI

# Page configuration
st.set_page_config(
    page_title="Advanced OCR & Document Understanding",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .ocr-engine {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bbdefb;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = {}
if 'document_type' not in st.session_state:
    st.session_state.document_type = "Unknown"

def classify_document(text):
    """Simple document classification based on keywords"""
    text_lower = text.lower()
    
    # Document type keywords
    document_keywords = {
        "Invoice": ["invoice", "bill", "amount due", "total amount", "payment terms"],
        "Receipt": ["receipt", "purchase", "total", "tax", "change", "cash"],
        "ID Document": ["passport", "driver license", "id card", "national id", "identity"],
        "Business Card": ["phone", "email", "@", "www", "company", "title"],
        "Contract": ["agreement", "contract", "terms", "conditions", "party"],
        "Medical Document": ["patient", "diagnosis", "treatment", "prescription", "doctor"],
        "Financial Statement": ["balance", "income", "expense", "account", "statement"]
    }
    
    scores = {}
    for doc_type, keywords in document_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        scores[doc_type] = score
    
    # Return the document type with highest score
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return "Unknown"

def extract_key_fields(text, doc_type):
    """Extract key information fields based on document type"""
    fields = {}
    text_lower = text.lower()
    
    if doc_type == "Invoice":
        # Extract invoice fields
        import re
        
        # Amount patterns
        amount_patterns = [
            r'total[:\s]*\$?([\d,]+\.?\d*)',
            r'amount[:\s]*\$?([\d,]+\.?\d*)',
            r'\$([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*dollars?'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text_lower)
            if match:
                fields['Total Amount'] = match.group(1)
                break
        
        # Date patterns
        date_patterns = [
            r'date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text_lower)
            if match:
                fields['Date'] = match.group(1)
                break
    
    elif doc_type == "Receipt":
        # Extract receipt fields
        import re
        
        # Store name (usually at the top)
        lines = text.split('\n')
        if lines:
            fields['Store Name'] = lines[0].strip()
        
        # Total amount (prioritize final TOTAL over SUBTOTAL)
        total_match = re.search(r'(?<!sub)total[:\s]*\$?([\d,]+\.?\d*)', text_lower)
        if total_match:
            fields['Total'] = total_match.group(1)
        else:
            amount_patterns = [
                r'total[:\s]*\$?([\d,]+\.?\d*)',
                r'\$([\d,]+\.?\d*)',
            ]
            for pattern in amount_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    fields['Total'] = match.group(1)
                    break
    
    elif doc_type == "Business Card":
        # Extract contact information
        import re
        
        # Email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            fields['Email'] = email_match.group()
        
        # Phone
        phone_match = re.search(r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})', text)
        if phone_match:
            fields['Phone'] = ''.join(phone_match.groups())
        
        # Name (usually first line)
        lines = text.split('\n')
        if lines:
            fields['Name'] = lines[0].strip()
    
    return fields

def ocr_tesseract(image):
    """OCR using Tesseract"""
    try:
        # Get text
        text = pytesseract.image_to_string(image)
        
        # Get bounding boxes
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Extract words with bounding boxes
        words = []
        for i, conf in enumerate(data['conf']):
            if conf > 0:  # Filter out low confidence results
                word = data['text'][i]
                if word.strip():
                    words.append({
                        'text': word,
                        'confidence': conf / 100,
                        'bbox': [data['left'][i], data['top'][i], 
                                data['width'][i], data['height'][i]]
                    })
        
        return {
            'text': text,
            'words': words,
            'confidence': np.mean([w['confidence'] for w in words]) if words else 0
        }
    except Exception as e:
        return {'text': f"Tesseract Error: {str(e)}", 'words': [], 'confidence': 0}

def ocr_easyocr(image):
    """OCR using EasyOCR"""
    try:
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Perform OCR
        results = reader.readtext(image_np)
        
        # Extract text and bounding boxes
        text = '\n'.join([result[1] for result in results])
        words = []
        
        for bbox, text_word, conf in results:
            # Convert bbox to [x, y, width, height] format
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x, y = min(x_coords), min(y_coords)
            width = max(x_coords) - x
            height = max(y_coords) - y
            
            words.append({
                'text': text_word,
                'confidence': conf,
                'bbox': [int(x), int(y), int(width), int(height)]
            })
        
        return {
            'text': text,
            'words': words,
            'confidence': np.mean([w['confidence'] for w in words]) if words else 0
        }
    except Exception as e:
        return {'text': f"EasyOCR Error: {str(e)}", 'words': [], 'confidence': 0}

def ocr_doctr(image):
    """OCR using docTR"""
    if not DOCTR_AVAILABLE:
        return {'text': "doctr not available", 'words': [], 'confidence': 0}
    
    try:
        # Convert PIL to file path or bytes
        if isinstance(image, Image.Image):
            # Save to temporary file
            temp_path = "temp_image.jpg"
            image.save(temp_path)
            doc = DocumentFile.from_images(temp_path)
            os.remove(temp_path)
        else:
            doc = DocumentFile.from_images(image)
        
        predictor = ocr_predictor(pretrained=True)
        result = predictor(doc)
        
        # Extract text and bounding boxes
        text = result.render()
        words = []
        
        # Extract words from result
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        words.append({
                            'text': word.value,
                            'confidence': word.confidence,
                            'bbox': [word.geometry[0][0], word.geometry[0][1],
                                   word.geometry[1][0] - word.geometry[0][0],
                                   word.geometry[1][1] - word.geometry[0][1]]
                        })
        
        return {
            'text': text,
            'words': words,
            'confidence': np.mean([w['confidence'] for w in words]) if words else 0
        }
    except Exception as e:
        return {'text': f"docTR Error: {str(e)}", 'words': [], 'confidence': 0}

def ocr_gemini(image, prompt):
    """OCR using Google Gemini"""
    if not GEMINI_AVAILABLE:
        return {'text': "Gemini not available", 'words': [], 'confidence': 0}
    
    try:
        # Configure Gemini
        # Check multiple possible environment variable names
        api_key = (os.getenv("GOOGLE_API_KEY") or 
                   os.getenv("GEMINI_API_KEY") or 
                   os.getenv("GOOGLE_GEMINI_API_KEY"))
        if not api_key:
            return {'text': "Google API key not configured. Please set GOOGLE_API_KEY, GEMINI_API_KEY, or GOOGLE_GEMINI_API_KEY in your environment variables", 'words': [], 'confidence': 0}
        
        genai.configure(api_key=api_key)
        # Lowest cost multimodal model supporting vision OCR: gemini-3.5-flash-lite
        try:
            model = genai.GenerativeModel("gemini-3.5-flash-lite")
            response = model.generate_content([prompt, image])
        except Exception:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content([prompt, image])
        
        return {
            'text': response.text,
            'words': [],  # Gemini doesn't provide bounding boxes
            'confidence': 0.9  # Assume high confidence for Gemini
        }
    except Exception as e:
        return {'text': f"Gemini Error: {str(e)}", 'words': [], 'confidence': 0}

def draw_bounding_boxes(image, words, engine_name):
    """Draw bounding boxes on image"""
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()
    
    # Convert to BGR for OpenCV
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_np
    
    # Draw bounding boxes
    for word in words:
        if 'bbox' in word and len(word['bbox']) == 4:
            try:
                x, y, w, h = word['bbox']
                # Convert to integers and ensure they're valid
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Check if coordinates are within image bounds
                img_height, img_width = img_bgr.shape[:2]
                if x >= 0 and y >= 0 and x + w <= img_width and y + h <= img_height:
                    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Only add text if there's enough space
                    if y - 15 >= 0:
                        cv2.putText(img_bgr, word['text'], (x, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except (ValueError, TypeError) as e:
                # Skip invalid bounding boxes
                continue
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def debug_bounding_boxes(words, engine_name):
    """Debug function to understand bounding box data"""
    if not words:
        return "No words detected"
    
    debug_info = f"**{engine_name} Bounding Box Debug:**\n"
    debug_info += f"Total words: {len(words)}\n\n"
    
    for i, word in enumerate(words[:5]):  # Show first 5 words
        if 'bbox' in word:
            bbox = word['bbox']
            debug_info += f"Word {i+1}: '{word.get('text', 'N/A')}'\n"
            debug_info += f"  Bbox: {bbox} (type: {type(bbox)})\n"
            debug_info += f"  Length: {len(bbox) if hasattr(bbox, '__len__') else 'N/A'}\n"
            if len(bbox) == 4:
                try:
                    x, y, w, h = bbox
                    debug_info += f"  x={x} (type: {type(x)}), y={y} (type: {type(y)})\n"
                    debug_info += f"  w={w} (type: {type(w)}), h={h} (type: {type(h)})\n"
                except Exception as e:
                    debug_info += f"  Error parsing bbox: {e}\n"
            debug_info += "\n"
    
    return debug_info

def download_results(results, filename="ocr_results.json"):
    """Create downloadable JSON file"""
    # Convert PIL images to base64 for JSON serialization
    serializable_results = {}
    for engine, result in results.items():
        serializable_results[engine] = {
            'text': result.get('text', ''),
            'confidence': result.get('confidence', 0),
            'words_count': len(result.get('words', [])),
            'words': result.get('words', [])
        }
    
    # Add document classification
    if 'document_type' in st.session_state:
        serializable_results['document_classification'] = st.session_state.document_type
    
    # Create JSON file
    json_str = json.dumps(serializable_results, indent=2)
    
    # Create download button
    st.download_button(
        label="📥 Download Results (JSON)",
        data=json_str,
        file_name=filename,
        mime="application/json"
    )

def main():
    # Header
    st.markdown('<h1 class="main-header">📄 Advanced OCR & Document Understanding</h1>', unsafe_allow_html=True)
    st.markdown("**Multi-Engine OCR with Document Classification, Key Information Extraction, and AI-Powered Insights**")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # OCR Engine Selection
    st.sidebar.subheader("🔧 OCR Engines")
    
    # Show engine availability status
    st.sidebar.markdown("**Engine Status:**")
    tesseract_status = "✅ Available" if True else "❌ Not Available"
    easyocr_status = "✅ Available" if True else "❌ Not Available"
    doctr_status = "✅ Available" if DOCTR_AVAILABLE else "❌ Not Available"
    gemini_status = "✅ Available" if GEMINI_AVAILABLE else "❌ Not Available"
    
    st.sidebar.markdown(f"• Tesseract: {tesseract_status}")
    st.sidebar.markdown(f"• EasyOCR: {easyocr_status}")
    st.sidebar.markdown(f"• docTR: {doctr_status}")
    st.sidebar.markdown(f"• Gemini: {gemini_status}")
    
    if not DOCTR_AVAILABLE:
        st.sidebar.info("💡 Install doctr: `pip install python-doctr[torch,viz]`")
    if not GEMINI_AVAILABLE:
        st.sidebar.info("💡 Install Gemini: `pip install google-generativeai`")
    
    st.sidebar.markdown("---")
    
    engines = {
        "Tesseract": st.sidebar.checkbox("Tesseract", value=True),
        "EasyOCR": st.sidebar.checkbox("EasyOCR", value=True),
        "docTR": st.sidebar.checkbox("docTR", value=DOCTR_AVAILABLE, disabled=not DOCTR_AVAILABLE),
        "Gemini": st.sidebar.checkbox("Gemini", value=GEMINI_AVAILABLE, disabled=not GEMINI_AVAILABLE)
    }
    
    # Gemini prompt
    gemini_prompt = st.sidebar.text_area(
        "🤖 Gemini Prompt",
        value="Perform OCR on this image. Transcribe all text you can see, preserving the original line breaks and formatting as much as possible. Do not add any extra comments or explanations.",
        height=100
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show detailed debugging information")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload Document", "📊 Analysis Results", "📈 Comparison & Insights"])
    
    with tab1:
        st.subheader("📤 Upload Document")
        
        # Quick sample loader with sample multi-page PDF
        sample_pdf_path = os.path.join(os.path.dirname(__file__), "sample_docs", "sample_document.pdf")
        if os.path.exists(sample_pdf_path):
            col_sample, _ = st.columns([1, 2])
            with col_sample:
                if st.button("📄 Load Sample Multi-Page Document (PDF)", help="Click to test with preloaded multi-page document"):
                    st.session_state['sample_pdf_loaded'] = True
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image or PDF file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf'],
            help="Upload a document to analyze"
        )
        
        pages = []
        if uploaded_file is not None:
            if uploaded_file.type.startswith('image'):
                img = Image.open(uploaded_file).convert('RGB')
                pages.append(("Image 1", img))
                st.image(img, caption="Uploaded Image")
            elif uploaded_file.type == 'application/pdf':
                pdf_bytes = uploaded_file.read()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                total_pages = len(pdf_document)
                st.info(f"📄 PDF has **{total_pages} page(s)**.")
                
                for p_idx in range(total_pages):
                    page = pdf_document[p_idx]
                    pix = page.get_pixmap(dpi=150)
                    p_img = Image.open(BytesIO(pix.tobytes("png"))).convert('RGB')
                    pages.append((f"Page {p_idx + 1}", p_img))
                
                # Show thumbnail previews
                cols = st.columns(min(len(pages), 4))
                for idx, (p_name, p_img) in enumerate(pages):
                    with cols[idx % 4]:
                        st.image(p_img, caption=p_name, use_container_width=True)
                        
        elif st.session_state.get('sample_pdf_loaded', False) and os.path.exists(sample_pdf_path):
            pdf_document = fitz.open(sample_pdf_path)
            total_pages = len(pdf_document)
            st.info(f"📄 Loaded Sample PDF with **{total_pages} page(s)**.")
            for p_idx in range(total_pages):
                page = pdf_document[p_idx]
                pix = page.get_pixmap(dpi=150)
                p_img = Image.open(BytesIO(pix.tobytes("png"))).convert('RGB')
                pages.append((f"Page {p_idx + 1}", p_img))
            
            # Show thumbnail previews
            cols = st.columns(min(len(pages), 4))
            for idx, (p_name, p_img) in enumerate(pages):
                with cols[idx % 4]:
                    st.image(p_img, caption=p_name, use_container_width=True)
            
        if pages:
            # Process button
            if st.button(f"🔍 Process All {len(pages)} Page(s)", type="primary"):
                progress_bar = st.progress(0)
                all_page_results = {}
                
                with st.spinner("Processing pages with selected OCR engines..."):
                    for p_i, (p_name, p_img) in enumerate(pages):
                        page_results = {}
                        for engine_name, enabled in engines.items():
                            if enabled:
                                if engine_name == "Tesseract":
                                    page_results[engine_name] = ocr_tesseract(p_img)
                                elif engine_name == "EasyOCR":
                                    page_results[engine_name] = ocr_easyocr(p_img)
                                elif engine_name == "docTR":
                                    page_results[engine_name] = ocr_doctr(p_img)
                                elif engine_name == "Gemini":
                                    page_results[engine_name] = ocr_gemini(p_img, gemini_prompt)
                        
                        all_page_results[p_name] = {
                            "image": p_img,
                            "results": page_results
                        }
                        progress_bar.progress((p_i + 1) / len(pages))
                
                st.session_state['all_page_results'] = all_page_results
                # Keep backward compatibility for single-page summary
                first_p = next(iter(all_page_results.values()))
                st.session_state.ocr_results = first_p['results']
                if first_p['results']:
                    st.session_state.document_type = classify_document(next(iter(first_p['results'].values()))['text'])
                st.success(f"✅ Successfully processed {len(pages)} page(s)!")
    
    with tab2:
        st.subheader("📊 Analysis Results")
        
        all_results = st.session_state.get('all_page_results', None)
        if all_results:
            page_names = list(all_results.keys())
            selected_page_name = st.radio("📑 Select Page to View:", page_names, horizontal=True)
            active_page_data = all_results[selected_page_name]
            active_page_image = active_page_data["image"]
            active_ocr_results = active_page_data["results"]
            
            # Document classification
            if active_ocr_results:
                first_txt = next(iter(active_ocr_results.values()))['text']
                detected_type = classify_document(first_txt)
                st.markdown(f"**📋 Document Type for {selected_page_name}:** `{detected_type}`")
            
            # OCR Results per engine for selected page
            for engine_name, result in active_ocr_results.items():
                with st.expander(f"🔍 {engine_name} Results ({selected_page_name})", expanded=True):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**📝 Extracted Text:**")
                        st.text_area(
                            f"{engine_name} Text ({selected_page_name})",
                            value=result['text'],
                            height=250,
                            key=f"text_{engine_name}_{selected_page_name}"
                        )
                        
                        # Key information extraction
                        if result['text']:
                            fields = extract_key_fields(result['text'], detected_type)
                            if fields:
                                st.markdown("**🔑 Key Information:**")
                                for field, value in fields.items():
                                    st.write(f"**{field}:** {value}")
                    
                    with col2:
                        st.markdown("**📊 Statistics:**")
                        col_metric1, col_metric2 = st.columns(2)
                        with col_metric1:
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                        with col_metric2:
                            word_count = len(result['words']) if result['words'] else len(result['text'].split())
                            st.metric("Words Detected", word_count)
                        
                        # Bounding boxes visualization
                        if result['words']:
                            st.markdown("**🎯 Text Detection (Bounding Boxes):**")
                            try:
                                annotated_image = draw_bounding_boxes(active_page_image, result['words'], engine_name)
                                st.image(annotated_image, caption=f"{engine_name} Detection - {selected_page_name}")
                            except Exception as e:
                                st.warning(f"Could not draw bounding boxes for {engine_name}: {str(e)}")
                                st.image(active_page_image, caption=f"Original {selected_page_name}")
                        else:
                            st.caption("ℹ️ *Gemini Vision transcribes full semantic markdown directly without pixel bounding box coordinates.*")
            
            # Download results
            download_results(active_ocr_results, filename=f"ocr_{selected_page_name.lower().replace(' ', '_')}.json")
        else:
            st.info("👆 Upload a document and click 'Process All Page(s)' to see results.")
    
    with tab3:
        st.subheader("📈 Comparison & Insights")
        
        if st.session_state.ocr_results:
            # Engine comparison
            st.markdown("**🔍 OCR Engine Comparison**")
            
            # Create comparison metrics
            comparison_data = []
            for engine_name, result in st.session_state.ocr_results.items():
                comparison_data.append({
                    'Engine': engine_name,
                    'Confidence': result['confidence'],
                    'Words Detected': len(result['words']),
                    'Text Length': len(result['text'])
                })
            
            # Display metrics
            cols = st.columns(len(comparison_data))
            for i, data in enumerate(comparison_data):
                with cols[i]:
                    st.markdown(f"**{data['Engine']}**")
                    st.metric("Confidence", f"{data['Confidence']:.2%}")
                    st.metric("Words", data['Words Detected'])
                    st.metric("Characters", data['Text Length'])
            
            # Text similarity analysis
            st.markdown("**📊 Text Similarity Analysis**")
            
            # Compare texts from different engines
            texts = {name: result['text'] for name, result in st.session_state.ocr_results.items()}
            
            if len(texts) > 1:
                # Simple similarity (word overlap)
                engine_names = list(texts.keys())
                for i in range(len(engine_names)):
                    for j in range(i + 1, len(engine_names)):
                        engine1, engine2 = engine_names[i], engine_names[j]
                        text1, text2 = texts[engine1], texts[engine2]
                        
                        # Calculate similarity
                        words1 = set(text1.lower().split())
                        words2 = set(text2.lower().split())
                        
                        if words1 or words2:
                            similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                            
                            st.write(f"**{engine1} vs {engine2}:** {similarity:.2%} similarity")
                            st.progress(similarity)
        else:
            st.info("👆 Process a document to see comparison and insights.")

if __name__ == "__main__":
    main() 