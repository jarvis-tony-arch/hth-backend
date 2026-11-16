import re
import nltk
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nlp_spacy = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
vader = SentimentIntensityAnalyzer()

STOP_WORDS = stopwords.words('english')
KEEP_WORDS = {'not', 'no', 'never', 'twice', 'double',
              'wrong', 'cannot', 'cant', "can't", 'unable'}
STOP_WORDS = [w for w in STOP_WORDS if w not in KEEP_WORDS]

# ─── Hinglish normalization map ───────────────────────
HINGLISH_MAP = [
    # Order/delivery — phrase level first
    (r'\border\s*(nhi|nahi|nahin|nai)\s*(aya|aaya|aai|mila)', 'order not arrived'),
    (r'\bdelivery\s*(nhi|nahi|nahin|nai)\s*(aya|aaya|hui|mili)', 'delivery not arrived'),
    (r'\bpackage\s*(nhi|nahi|nahin|nai)\s*(aya|aaya|mila)', 'package not arrived'),
    (r'\babhi\s*(tak|bhi)\s*(nhi|nahi)\s*(aya|aaya|mila)', 'still not arrived'),
    (r'\bmera\s*order\b', 'my order'),

    # Payment/charge — phrase level
    (r'\bpaise\s*(do|2|dono)\s*(baar|bar|times?)\s*(kat[ea]?|gaye|gaya|deduct)', 'charged twice'),
    (r'\b(do|2|dono)\s*(baar|bar|times?)\s*(charge|payment|kat[ea]?|deduct)', 'charged twice'),
    (r'\bpaise\s*(kat\s*gaye|kat\s*gaya|cut\s*ho\s*gaye)', 'money deducted'),
    (r'\bdouble\s*(charge|payment|deduction|kat)', 'charged twice'),
    (r'\bpayment\s*(fail|nahi|nhi)\s*(hua|hui|ho\s*gaya)', 'payment failed'),
    (r'\bdo\s*baar\s*(charge|paise|payment)', 'charged twice'),

    # Refund
    (r'\b(wapas|vapas|refund)\s*(chahiye|karo|karna|do)', 'need refund'),
    (r'\bpaise\s*(wapas|vapas)\s*(chahiye|karo|do)', 'money back refund'),

    # Account
    (r'\b(login|account)\s*(nhi|nahi|nahin)\s*(ho|kar)\s*(raha|rahi|pa)', 'cannot login'),
    (r'\baccount\s*(band|block|lock)\s*(ho\s*gaya|hua)', 'account locked'),
    (r'\bpassword\s*(bhool|forgot|yaad\s*nahi)', 'forgot password'),

    # Wrong/damaged item
    (r'\bgalat\s*(item|product|samaan)', 'wrong item received'),
    (r'\btoota\s*(hua|item|product)', 'damaged item'),
    (r'\bkharab\s*(item|product|samaan)', 'damaged item'),

    # Word level replacements
    (r'\bnhi\b', 'not'),
    (r'\bnahi\b', 'not'),
    (r'\bnai\b', 'not'),
    (r'\bnahin\b', 'not'),
    (r'\bhua\b', ''),
    (r'\bhui\b', ''),
    (r'\bkarna\b', ''),
    (r'\bhai\b', 'is'),
    (r'\btha\b', 'was'),
    (r'\bpaise\b', 'money payment'),
    (r'\brupaye\b', 'rupees'),
    (r'\bkaat\b', 'deducted'),
    (r'\bkata\b', 'deducted'),
    (r'\bkate\b', 'deducted'),
    (r'\bkat\b', 'deducted'),
    (r'\bbaar\b', 'times'),
    (r'\bbar\b', 'times'),
    (r'\baaya\b', 'arrived'),
    (r'\baya\b', 'arrived'),
    (r'\bmila\b', 'received'),
    (r'\bgalat\b', 'wrong'),
    (r'\bsamaan\b', 'item product'),
    (r'\bturant\b', 'immediately urgent'),
    (r'\bjaldi\b', 'quickly urgent'),
    (r'\babhi\b', 'immediately'),
    (r'\bmera\b', 'my'),
    (r'\bmeri\b', 'my'),
    (r'\bmujhe\b', 'me'),
    (r'\bhumara\b', 'our'),
    (r'\baapka\b', 'your'),
    (r'\bkaro\b', 'do'),
    (r'\bchahiye\b', 'needed'),
    (r'\bwapas\b', 'back'),
    (r'\bvapas\b', 'back'),
    (r'\bband\b', 'blocked'),
    (r'\bblock\b', 'blocked'),
]


def normalize_hinglish(text: str) -> str:
    """Convert Hinglish (Hindi in Roman script) to English."""
    result = text.lower()
    for pattern, replacement in HINGLISH_MAP:
        result = re.sub(pattern, replacement, result,
                        flags=re.IGNORECASE)
    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def is_hinglish(text: str) -> bool:
    """Detect if text contains Hinglish markers."""
    hinglish_markers = [
        r'\b(mera|meri|mujhe|humara|aapka)\b',
        r'\b(nhi|nahi|nahin|nai)\b',
        r'\b(hua|hui|hoga|hogi|gaya|gaye)\b',
        r'\b(baar|karo|chahiye|wapas|vapas)\b',
        r'\b(aaya|aya|mila)\b',
        r'\b(paise|rupaye|paisa)\b',
        r'\b(abhi|jaldi|turant)\b',
        r'\b(galat|sahi|kharab|toota)\b',
    ]
    text_lower = text.lower()
    matches = sum(
        1 for p in hinglish_markers
        if re.search(p, text_lower, re.IGNORECASE)
    )
    return matches >= 2


# ─── Intent taxonomy ──────────────────────────────────
INTENT_PATTERNS = {
    'delivery_issue': [
        r'\border.{0,30}(not arrived?|not delivered?|not received?|not here|missing|delayed?|late)\b',
        r'\b(havent|haven\'t|didnt|didn\'t|have not|did not).{0,20}(received?|got)\b.{0,20}\border\b',
        r'\b(where is|track).{0,20}(order|package|parcel)\b',
        r'\b(order|package|parcel|shipment).{0,20}(not arrived?|missing|lost|delayed?)\b',
        r'\bstill (waiting|not received?|not arrived?)\b',
        r'\b(not arrived?|not delivered?|not received?|not here)\b',
        r'\border.{0,20}arrived\b',
    ],
    'duplicate_payment': [
        r'\bcharged?.{0,20}(twice|two times|double|again|duplicate)\b',
        r'\b(twice|double|duplicate).{0,20}(charged?|deducted?|billed?|payment)\b',
        r'\bduplicate (payment|charge|transaction)\b',
        r'\b(two|2).{0,10}(payment|charge|transaction|deduction|times)\b',
        r'\bpaid (twice|double|two times)\b',
        r'\bmoney.{0,20}deducted.{0,20}twice\b',
        r'\bcharged twice\b',
    ],
    'refund_request': [
        r'\b(refund|money back|reimburse|get my money back)\b',
        r'\b(reverse|cancel).{0,20}(payment|transaction|charge)\b',
        r'\bwant.{0,20}refund\b',
        r'\bneed.{0,20}refund\b',
        r'\bmoney.{0,20}(back|return|wapas)\b',
    ],
    'account_access': [
        r'\b(cant|can\'t|cannot|unable to|can not).{0,20}(login|log in|signin|sign in|access)\b',
        r'\b(account|login).{0,20}(locked|blocked|suspended|disabled)\b',
        r'\b(forgot|reset|change|lost).{0,20}password\b',
        r'\bpassword.{0,20}(forgot|reset|change|lost|expired)\b',
        r'\bcannot login\b',
    ],
    'unauthorized_payment': [
        r'\b(unauthorized|unknown|suspicious|fraud|fraudulent|not authorized)\b.{0,30}(payment|transaction|charge)\b',
        r'\b(someone|somebody).{0,30}(used|made|placed|charged?|hacked?)\b',
        r'\bmy (account|card).{0,20}(hacked?|compromised?|stolen?)\b',
    ],
    'wrong_item': [
        r'\b(wrong|incorrect|different).{0,20}(item|product|order|received?)\b',
        r'\b(received?|got|delivered?).{0,30}(wrong|incorrect|different)\b',
        r'\bsent (me )?(wrong|different|incorrect)\b',
        r'\bwrong item received\b',
    ],
    'damaged_item': [
        r'\b(damaged?|broken?|defective?|faulty|cracked?|not working)\b.{0,30}(item|product|order)\b',
        r'\b(item|product|order).{0,30}(damaged?|broken?|defective?|faulty)\b',
        r'\barrived? (damaged?|broken?|cracked?|defective?)\b',
        r'\bdamaged item\b',
    ],
    'payment_failure': [
        r'\b(payment|transaction).{0,20}(failed?|declined?|not processed?|unsuccessful)\b',
        r'\b(could not|couldnt|unable to|can not).{0,20}(pay|payment|checkout)\b',
        r'\bpayment (not|didnt|did not) (go through|work|process)\b',
        r'\bpayment failed\b',
    ],
    'app_technical': [
        r'\b(app|application|website|site).{0,20}(not working|crashed?|down|error|bug|slow)\b',
        r'\bkeeps? (crashing?|freezing?|logging? out)\b',
    ],
    'order_cancellation': [
        r'\bcancel.{0,20}order\b',
        r'\b(want|need).{0,20}cancel.{0,20}order\b',
    ],
    'order_status': [
        r'\b(status|update|where).{0,20}(order|package|delivery)\b',
        r'\b(when|how long).{0,20}(deliver|arrive|ship)\b',
        r'\btrack.{0,20}(order|package|shipment)\b',
    ],
}

DEPARTMENT_MAP = {
    'delivery_issue':       'Logistics',
    'duplicate_payment':    'Payments',
    'refund_request':       'Refunds',
    'account_access':       'Account Security',
    'unauthorized_payment': 'Fraud & Security',
    'wrong_item':           'Order Management',
    'damaged_item':         'Order Management',
    'payment_failure':      'Payments',
    'app_technical':        'Technical Support',
    'order_cancellation':   'Order Management',
    'order_status':         'Logistics',
}

ISSUE_DISPLAY = {
    'delivery_issue':       'Delivery Issue',
    'duplicate_payment':    'Duplicate Payment',
    'refund_request':       'Refund Request',
    'account_access':       'Account Access',
    'unauthorized_payment': 'Unauthorized Payment',
    'wrong_item':           'Wrong Item Received',
    'damaged_item':         'Damaged Item',
    'payment_failure':      'Payment Failure',
    'app_technical':        'Technical Problem',
    'order_cancellation':   'Order Cancellation',
    'order_status':         'Order Status Inquiry',
}

CRITICAL_PATTERNS = [
    r'\b(hack|hacked|unauthorized|fraud|fraudulent|stolen|compromised)\b',
]

HIGH_PATTERNS = [
    r'\b(immediately|urgent|asap|right now|emergency|quickly|right away)\b',
    r'\b(charged?|deducted?).{0,20}(twice|double|duplicate)\b',
    r'\b(money|payment|amount).{0,20}(lost|missing|wrong|deducted?)\b',
]

LOW_PATTERNS = [
    r'\b(information|question|curious|wondering|just wanted|would like to know)\b',
]


def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't|cannot", "can not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_entities(text: str) -> dict:
    entities = {}

    order_match = re.search(
        r'#?\b([A-Z]{0,3}\d{4,10})\b|order\s*(id|number|no|#)?\s*[:#]?\s*([A-Z0-9]{4,12})',
        text, re.IGNORECASE
    )
    if order_match:
        eid = order_match.group(1) or order_match.group(3)
        if eid:
            entities['Order ID'] = f'#{eid}'

    amount_match = re.search(
        r'[₹$€£]\s*[\d,]+(?:\.\d{1,2})?|[\d,]+(?:\.\d{1,2})?\s*(?:rupees?|rs\.?|inr|dollars?)',
        text, re.IGNORECASE
    )
    if amount_match:
        entities['Amount'] = amount_match.group(0).strip()

    payment_match = re.search(
        r'\b(credit card|debit card|card|upi|net banking|paypal|wallet|gpay|phonepe|paytm|google pay)\b',
        text, re.IGNORECASE
    )
    if payment_match:
        entities['Payment Method'] = payment_match.group(0).title()

    doc = nlp_spacy(text)
    for ent in doc.ents:
        if ent.label_ == 'PRODUCT' and 'Product' not in entities:
            entities['Product'] = ent.text.title()
        elif ent.label_ == 'DATE' and 'Date' not in entities:
            entities['Date'] = ent.text
        elif ent.label_ == 'ORG' and 'Merchant' not in entities:
            entities['Merchant'] = ent.text.title()

    return entities


def detect_issues(text: str) -> list:
    text_lower = preprocess(text)
    found = []

    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 1
        if score > 0:
            confidence = min(0.65 + (score * 0.12), 0.97)
            found.append({
                'intent': intent,
                'display': ISSUE_DISPLAY[intent],
                'department': DEPARTMENT_MAP[intent],
                'confidence': round(confidence, 2),
                'matched_count': score,
            })

    found.sort(key=lambda x: x['confidence'], reverse=True)
    return found


def detect_urgency(text: str, issues: list) -> str:
    text_lower = preprocess(text)

    for p in CRITICAL_PATTERNS:
        if re.search(p, text_lower, re.IGNORECASE):
            return 'CRITICAL'

    financial = {'duplicate_payment', 'unauthorized_payment',
                 'refund_request', 'payment_failure'}
    issue_set = {i['intent'] for i in issues}
    if len(issue_set & financial) >= 2:
        return 'HIGH'

    for p in HIGH_PATTERNS:
        if re.search(p, text_lower, re.IGNORECASE):
            return 'HIGH'

    if len(issues) >= 3:
        return 'HIGH'
    if len(issues) >= 2:
        return 'MEDIUM'

    for p in LOW_PATTERNS:
        if re.search(p, text_lower, re.IGNORECASE):
            return 'LOW'

    return 'MEDIUM'


def detect_sentiment(text: str) -> str:
    scores = vader.polarity_scores(text)
    compound = scores['compound']

    frustrated = r'\b(frustrated?|angry|furious|terrible|horrible|worst|unacceptable|ridiculous)\b'
    if re.search(frustrated, text, re.IGNORECASE):
        return 'Frustrated'
    if compound <= -0.5:
        return 'Very Negative'
    if compound <= -0.1:
        return 'Negative'
    if compound >= 0.1:
        return 'Positive'
    return 'Neutral'


def generate_xai(issues: list, urgency: str,
                 confidence: float) -> list:
    reasons = []
    for issue in issues:
        reasons.append(
            f"{issue['department']} assigned — '{issue['display']}' "
            f"detected with {int(issue['confidence'] * 100)}% confidence"
        )
    reasons.append(
        f"{urgency} urgency — based on issue severity and financial impact"
    )
    if confidence >= 0.65:
        reasons.append(
            f"Auto-routing applied — confidence {int(confidence * 100)}% "
            f"exceeds 65% threshold"
        )
    else:
        reasons.append(
            f"Human review flagged — confidence {int(confidence * 100)}% "
            f"below 65% threshold"
        )
    return reasons


def detect_missing_info(issues: list, entities: dict) -> list:
    missing = []
    intents = {i['intent'] for i in issues}

    if intents & {'delivery_issue', 'order_status',
                  'wrong_item', 'damaged_item',
                  'order_cancellation'}:
        if 'Order ID' not in entities:
            missing.append('Order ID or order number')

    if intents & {'duplicate_payment', 'refund_request',
                  'unauthorized_payment', 'payment_failure'}:
        if 'Amount' not in entities:
            missing.append('Transaction amount')
        if 'Payment Method' not in entities:
            missing.append('Payment method used')

    return missing


def generate_title(issues: list) -> str:
    if not issues:
        return 'General Support Request'
    return ' + '.join([i['display'] for i in issues[:3]])


def generate_summary(text: str, issues: list,
                     entities: dict) -> str:
    issue_str = ', '.join(
        [i['display'].lower() for i in issues]
    ) if issues else 'general inquiry'

    parts = []
    if 'Order ID' in entities:
        parts.append(f"order {entities['Order ID']}")
    if 'Amount' in entities:
        parts.append(f"{entities['Amount']} charge")
    if 'Product' in entities:
        parts.append(entities['Product'])

    entity_str = (', involving ' + ', '.join(parts)) if parts else ''
    return (f"Customer reports {issue_str}"
            f"{entity_str}. Requires immediate attention.")


def analyze(text: str) -> dict:
    # Step 1 — Detect and normalize Hinglish
    hinglish_detected = is_hinglish(text)
    if hinglish_detected:
        processed_text = normalize_hinglish(text)
    else:
        processed_text = text

    # Step 2 — Detect issues on processed text
    issues = detect_issues(processed_text)

    # Step 3 — No issues fallback
    if not issues:
        return {
            'issues': [],
            'departments': [],
            'urgency': 'LOW',
            'sentiment': detect_sentiment(processed_text),
            'confidence': 0.0,
            'entities': {},
            'escalate': True,
            'ai_title': 'Unclear Request',
            'ai_summary': (
                'Unable to detect specific issues. '
                'Human review recommended.'
            ),
            'xai_reasons': [
                'No recognizable issue patterns found',
                'Routing to human review for clarification',
            ],
            'missing_info': [],
            'follow_up_questions': [
                'Could you describe your issue in more detail?',
                'What specific problem are you experiencing?',
            ],
            'hinglish_detected': hinglish_detected,
            'processed_text': processed_text,
        }

    # Step 4 — Extract everything else
    entities = extract_entities(processed_text)
    urgency = detect_urgency(processed_text, issues)
    sentiment = detect_sentiment(processed_text)
    confidence = round(
        sum(i['confidence'] for i in issues) / len(issues), 2
    )
    escalate = (confidence < 0.65 or urgency == 'CRITICAL')
    xai = generate_xai(issues, urgency, confidence)
    missing = detect_missing_info(issues, entities)

    # Step 5 — Follow-up questions
    follow_up = []
    if 'Order ID or order number' in missing:
        follow_up.append(
            'Could you please share your order ID or order number? '
            '(e.g. #45821)'
        )
    if 'Transaction amount' in missing:
        follow_up.append(
            'What was the amount that was charged? '
            '(e.g. ₹500, $20)'
        )
    if 'Payment method used' in missing:
        follow_up.append(
            'Which payment method did you use? '
            '(e.g. credit card, UPI, net banking)'
        )

    issue_intents = {i['intent'] for i in issues}

    if 'damaged_item' in issue_intents or 'wrong_item' in issue_intents:
        follow_up.append(
            'Could you share a photo of the item you received? '
            'This will help us resolve your case faster.'
        )
    if 'app_technical' in issue_intents:
        follow_up.append(
            'Which device and operating system are you using? '
            '(e.g. iPhone 13, Android 12)'
        )
    if 'account_access' in issue_intents:
        follow_up.append(
            'Please confirm the email address linked to your account.'
        )
    if 'unauthorized_payment' in issue_intents:
        follow_up.append(
            'When did you notice this unauthorized transaction? '
            'Please share the date and time if possible.'
        )

    return {
        'issues': issues,
        'departments': list({i['department'] for i in issues}),
        'urgency': urgency,
        'sentiment': sentiment,
        'confidence': confidence,
        'entities': entities,
        'escalate': escalate,
        'ai_title': generate_title(issues),
        'ai_summary': generate_summary(
            processed_text, issues, entities
        ),
        'xai_reasons': xai,
        'missing_info': missing,
        'follow_up_questions': follow_up,
        'hinglish_detected': hinglish_detected,
        'processed_text': processed_text,
    }