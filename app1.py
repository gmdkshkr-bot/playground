# --- 1. Gemini Analysis Function (Translated Prompt) ---
def analyze_receipt_with_gemini(_image: Image.Image):
    """
    Calls the Gemini model to extract data and categorize items from a receipt image.
    """
    
    prompt_template = """
    You are an expert in receipt analysis and ledger recording.
    Analyze the following items from the receipt image and **you must extract them in JSON format**.
    
    **CRITICAL INSTRUCTION:** The response must only contain the **JSON code block wrapped in backticks (```json)**. Do not include any explanations, greetings, or additional text outside the JSON code block.
    
    1. store_name: Store Name (text)
    2. date: Date (YYYY-MM-DD format). **If not found, use YYYY-MM-DD format based on today's date.**
    3. store_location: Store location/address (text). **If not found, use "Seoul".**
    4. total_amount: Final amount settled/paid via card or cash (numbers only, no commas). **CRITICAL: You MUST extract the FINAL '합계' (Total) amount settled by the customer, which reflects tax and discount.** 5. tax_amount: Tax or VAT amount recognized on the receipt (numbers only, no commas). Must be 0 if not present.
    6. tip_amount: Tip amount recognized on the receipt (numbers only, no commas). Must be 0 if not present.
    7. discount_amount: Total discount amount applied to the entire receipt (numbers only, no commas). **CRITICAL: Extract this as a POSITIVE number (e.g., if the discount is -18,000 KRW, output 18000). Must be 0 if not present.**
    8. currency_unit: Official currency code shown on the receipt (e.g., KRW, USD, EUR).
    9. items: List of purchased items. Each item must include:
        - name: Item Name (text)
        - price: Unit Price (numbers only, no commas). **This must be the final, VAT-INCLUSIVE price displayed next to the item name (before final discount allocation).** - quantity: Quantity (numbers only)
        - category: The most appropriate **Detailed Sub-Category** for this item, which must be **automatically classified** by you.
    
    **Classification Guide (Choose ONE sub-category for 'category' field):**
    - **FIXED / ESSENTIAL:** Rent & Mortgage, Communication Fees, Public Utilities, Public Transit, Fuel & Vehicle Maint., Parking & Tolls
    - **VARIABLE / CONSUMPTION (Planned):** Groceries, Household Goods
    - **VARIABLE / CONSUMPTION (Experience):** Dining Out, Travel & Accommodation, Movies & Shows
    - **INVESTMENT / ASSET:** Medical & Pharmacy, Health Supplements, Education & Books, Hobby & Skill Dev., Events & Gifts
    - **IMPULSE / LOSS:** Casual Dining, Coffee & Beverages, Alcohol & Bars, Games & Digital Goods, Taxi Convenience, Fees & Penalties, Unclassified
        
    JSON Schema:
    ```json
    {
      "store_name": "...",
      "date": "...",
      "store_location": "...",
      "total_amount": ...,
      "tax_amount": ...,
      "tip_amount": ...,
      "discount_amount": ...,
      "currency_unit": "...",  
      "items": [
        {"name": "...", "price": ..., "quantity": ..., "category": "..."}
      ]
    }
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_template, _image],
            config=genai.types.GenerateContentConfig(
                safety_settings=[
                    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                ]
            )
        )
        return response.text
    
    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        return None
# (이하 생략)
