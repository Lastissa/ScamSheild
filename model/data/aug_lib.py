"""
Helpers for making the synthetic scam dataset more realistic.

 * scam_url()   -> phishing-style links (lookalike domains, odd TLDs, shorteners)
 * legit URL    -> built from real official domains (or a company's own domain)
 * TEMPLATES    -> Nigerian-context benign + scam message families (English, Pidgin, es/fr/pt/hi)

NOTE: lookalike domains are generated from patterns and are NOT checked for registration.
Never open them; they exist only as training text.
"""
import random, re, string

rng = random.Random(2026)            # fixed seed -> the dataset is reproducible

# ----------------------------------------------------------------------------- random helpers
def digits(n):  return "".join(rng.choice("0123456789") for _ in range(n))
def slug(n=6):  return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(n))
def money(lo, hi, step):  return f"{rng.randrange(lo, hi, step):,}"

FIRST = ["Chinedu","Aisha","Tunde","Ngozi","Emeka","Funke","Ibrahim","Bisi","Kelechi","Hauwa","Segun",
         "Amaka","Yusuf","Folake","Chika","Musa","Adaeze","Femi","Zainab","Obinna","Temitope","Damilola"]

# ----------------------------------------------------------------------------- brands
# display name -> (official domain, lookalike base)   [official domains as publicly known]
BANKS  = {"GTBank": ("gtbank.com","gtbank"), "Zenith Bank": ("zenithbank.com","zenithbank"),
          "Access Bank": ("accessbankplc.com","accessbank"), "First Bank": ("firstbanknigeria.com","firstbank"),
          "UBA": ("ubagroup.com","uba"), "Wema Bank": ("wemabank.com","wemabank")}
TELCOS = {"MTN": ("mtn.ng","mtn"), "Airtel": ("airtel.com.ng","airtel"), "Glo": ("gloworld.com","glo")}
SHOPS  = {"Jumia": ("jumia.com.ng","jumia"), "DHL": ("dhl.com","dhl")}
GOVS   = {"NIMC": ("nimc.gov.ng","nimc"), "FRSC": ("frsc.gov.ng","frsc"), "JAMB": ("jamb.gov.ng","jamb")}
FINS   = {"Paystack": ("paystack.com","paystack")}
GLOBAL = {"Netflix": ("netflix.com/youraccount","netflix"), "PayPal": ("paypal.com/myaccount","paypal"),
          "Google": ("myaccount.google.com","google"), "Apple": ("appleid.apple.com","apple")}
SCHOOLS = {"University of Lagos": "unilag.edu.ng", "University of Ilorin": "unilorin.edu.ng",
           "University of Ibadan": "ui.edu.ng"}
OILCOS = ["NNPC","Shell","Chevron","Dangote","TotalEnergies"]
COS = [("Brightpath Logistics","brightpathlogistics.com"), ("Kola Foods","kolafoods.com.ng"),
       ("Adire Hub","adirehub.ng"), ("Lagos Fintech Labs","lagosfintechlabs.com"),
       ("Naija Fresh Market","naijafreshmarket.com.ng"), ("Sunrise Analytics","sunriseanalytics.io")]
INTL = [("BluePeak","bluepeak.com"), ("Mercado Sol","mercadosol.com"), ("Petit Marché","petitmarche.com")]
DISCOS = ["Ikeja Electric","Eko DisCo","IBEDC","Abuja DisCo"]

# ----------------------------------------------------------------------------- scam links
KW   = ["secure","verify","update","login","portal","support","alert","claim","reactivate","confirm","kyc","bvn","help"]
SCAM_TLDS = ["com"]*9 + ["net","net","xyz","top","online","site","click","info","live","icu","help","co","cc","vip","ng"]
SHORT = ["bit.ly","tinyurl.com","t.ly","cutt.ly","rb.gy","is.gd"]

def typo(s):
    """Look-alike spelling: homoglyph (o->0, l->1), swapped letters, or a doubled letter."""
    r = rng.randrange(3)
    if r == 0 and ("o" in s or "l" in s):
        return s.replace("o","0",1) if "o" in s else s.replace("l","1",1)
    if r == 1 and len(s) > 3:
        i = rng.randrange(1, len(s)-1); return s[:i] + s[i+1] + s[i] + s[i+2:]
    i = rng.randrange(len(s)); return s[:i] + s[i] + s[i:]

def lookalike(real_domain, base):
    kw, tld = rng.choice(KW), rng.choice(SCAM_TLDS)
    style = rng.randrange(6)
    if style == 0: return f"{base}-{kw}.{tld}"
    if style == 1: return f"{kw}-{base}.{tld}"
    if style == 2: return f"{typo(base)}.{rng.choice(['com','net','online'])}"
    if style == 3: return f"{base}{kw}.{tld}"
    if style == 4 and real_domain:      # classic trick: real name used as a decoy subdomain
        return f"{real_domain.split('/')[0]}.{kw}-{slug(4).lower()}.{tld}"
    return f"{base}-ng.{tld}"

def scam_url(real_domain=None, base=None):
    if rng.random() < 0.2:                                   # shortened link, often without scheme
        return rng.choice(["","https://"]) + f"{rng.choice(SHORT)}/{slug(7)}"
    dom = lookalike(real_domain, base) if base else f"{rng.choice(KW)}-{rng.choice(KW)}.{rng.choice(SCAM_TLDS)}"
    path = rng.choice(["", "/" + rng.choice(KW), "/" + rng.choice(KW) + "?id=" + slug(6)])
    return rng.choice(["https://","http://","http://",""]) + dom + path

LEGIT_TLDS = ["com"]*12 + ["com.ng"]*3 + ["ng","org","co","io","net"]
def legit_url(domain, path=""):
    """Legit links: mostly .com but also .com.ng/.ng/.org/etc., and ~8% use a shortener (real brands do too)."""
    if rng.random() < 0.08:
        return f"https://{rng.choice(SHORT)}/{slug(7)}"
    if domain.endswith(".com"):                       # fictional org names get a varied TLD
        domain = domain[:-4] + "." + rng.choice(LEGIT_TLDS)
    tail = "?utm_source=sms" if rng.random() < 0.25 else ""
    return f"https://{domain}{path}{tail}"

def org_slug(org):
    """'SecureTrust Bank' -> 'securetrustbank' (used to build an organisation's own domain)."""
    return re.sub(r"[^a-z0-9]", "", org.lower())[:24] or "company"

# ----------------------------------------------------------------------------- context for templates
def ctx(kind):
    c = dict(name=rng.choice(FIRST), name2=rng.choice(FIRST), amt=money(1000,250000,500), amt2=money(100,5000,100),
             amt3=money(500000,5000000,50000), bal=money(2000,900000,100), fee=money(2000,25000,500),
             n4=digits(4), n3=digits(3), otp=digits(6), ref=slug(8).upper(), meter=digits(11), units=f"{rng.uniform(20,300):.1f}",
             tok=f"{digits(4)}-{digits(4)}-{digits(4)}-{digits(4)}-{digits(4)}", gb=rng.choice([10,20,50,100]),
             num=rng.choice(["0803","0806","0810","0813","0701","0902","0816"]) + digits(7),
             date=rng.choice(["Friday","Monday","30 Nov","15 Dec","the end of this month","tomorrow"]),
             month=rng.choice(["October","November","September"]), n=rng.randrange(10,900), g=f"{slug(3)}-{slug(4)}-{slug(3)}".lower(),
             desc=rng.choice(["POS PURCHASE SHOPRITE IKEJA","NIP TRANSFER","AIRTIME PURCHASE","WEB PAYMENT NETFLIX","ATM WITHDRAWAL","POS PURCHASE FILLING STATION"]),
             role=rng.choice(["Data Analyst","DevOps Engineer","Customer Support Representative","Software Developer","Sales Executive"]),
             city=rng.choice(["Lagos","Abuja","Ibadan","Port Harcourt"]), merch=rng.choice(["Kola Foods","Adire Hub","Naija Fresh Market"]))
    pools = {"bank": BANKS, "telco": TELCOS, "shop": SHOPS, "gov": GOVS, "fin": FINS, "global": GLOBAL}
    if kind in pools:
        B = rng.choice(list(pools[kind])); real, base = pools[kind][B]
        c.update(B=B, D=real, URL=scam_url(real, base))
    elif kind == "uni":
        B = rng.choice(list(SCHOOLS)); c.update(B=B, D=SCHOOLS[B], URL=scam_url(SCHOOLS[B], org_slug(B)[:10]))
    elif kind == "co":
        B, D = rng.choice(COS); c.update(B=B, D=D, URL=scam_url(D, org_slug(B)))
    elif kind == "oil":
        B = rng.choice(OILCOS); c.update(B=B, D=org_slug(B)+".com", URL=scam_url(org_slug(B)+".com", org_slug(B)))
    elif kind == "disco":
        c.update(B=rng.choice(DISCOS), D="", URL=scam_url())
    elif kind == "intl":
        B, D = rng.choice(INTL); c.update(B=B, D=D, URL=scam_url(D, org_slug(B)))
    else:
        c.update(B="", D="", URL=scam_url())
    c["NAME2"] = c["name2"].upper()
    return c

# ----------------------------------------------------------------------------- templates
# family -> (label, language, kind, weight, [variants])      {X} fields are filled by ctx()
T = {}
def fam(name, label, lang, kind, weight, variants): T[name] = (label, lang, kind, weight, variants)

# ---------- BENIGN, Nigerian context (English)
fam("legit_ng_bank_debit", 0, "English", "bank", 5, [
 "{B}: Debit Alert. Amt: NGN {amt} Acct: ******{n4} Desc: {desc} Avail Bal: NGN {bal}. Not you? Visit {D} or call the number on your card.",
 "Transaction Alert [Debit] Amt: NGN {amt}.00 Acc: {n3}***{n4} Desc: {desc} Bal: NGN {bal}.00 {date}",
 "Hello {name}, your {B} account ending {n4} was debited NGN {amt} for {desc}. If you did not authorise this, block your card in the {B} app or visit your branch."])
fam("legit_ng_bank_credit", 0, "English", "bank", 4, [
 "{B}: Credit Alert. Amt: NGN {amt} Acct: ******{n4} Desc: TRANSFER FROM {NAME2} Avail Bal: NGN {bal}",
 "You have received NGN {amt} from {name2}. Your new balance is NGN {bal}. Thank you for banking with {B}."])
fam("legit_ng_bank_otp", 0, "English", "bank", 4, [
 "Your {B} OTP is {otp}. It expires in 10 minutes. Do not share this code with anyone, including {B} staff.",
 "{otp} is your verification code for {B}. Never share it. {B} will never ask for your OTP, PIN or password."])
fam("legit_ng_bank_statement", 0, "English", "bank", 3, [
 "{B}: your {month} account statement is ready. Log in at https://{D} to view or download it.",
 "Dear {name}, your {B} e-statement for {month} is available in the mobile app. You can also request it at https://{D}."])
fam("legit_ng_bank_scam_warning", 0, "English", "bank", 3, [
 "{B} reminder: we will never ask for your PIN, OTP or password by SMS, call or WhatsApp. Report suspicious messages at https://{D}.",
 "Beware of fraud. {B} staff will never ask you to click a link to unblock your account. Only use https://{D} or our official app.",
 "{B} security tip: do not share your card PIN, CVV or OTP with anyone. If someone calls claiming to be {B}, hang up and visit https://{D}.",
 "Dear customer, fraudsters are sending fake {B} messages. Our only website is https://{D}. Report scams to our contact centre."])
fam("legit_ng_telco_recharge", 0, "English", "telco", 4, [
 "You have successfully recharged NGN {amt2} airtime. Your {B} balance is NGN {bal}. Thank you for choosing {B}.",
 "{B}: your {gb}GB data bundle has been activated and is valid for 30 days. Check your balance in the {B} app."])
fam("legit_ng_telco_notice", 0, "English", "telco", 3, [
 "{B}: you have used 90% of your data bundle. Renew in the {B} app or at https://{D}.",
 "{B}: enjoy double data on your next top-up this weekend. Terms apply. Details at https://{D}.",
 "{B}: your SIM registration is complete. Thank you {name}. For help, visit https://{D} or any {B} service centre.",
 "{B}: your {gb}GB bundle expires on {date}. Auto-renew is on. Manage it in the {B} app."])
fam("legit_ng_meter_token", 0, "English", "disco", 3, [
 "Token: {tok} Meter: {meter} Units: {units} kWh Amt: NGN {amt} Thank you. {B}",
 "Your prepaid meter {meter} has been credited with {units} kWh. Token: {tok}. {B}"])
fam("legit_ng_shop_delivery", 0, "English", "shop", 4, [
 "{B}: your order {ref} is out for delivery today. Our agent will call you before arrival. Track in the app or at https://{D}.",
 "Hi {name}, your {B} order {ref} has shipped and should arrive by {date}. Pay on delivery: NGN {amt}. Track: https://{D}"])
fam("legit_ng_shop_promo", 0, "English", "shop", 3, [
 "{B}: Black Friday starts tonight! Up to 70% off phones, fashion and home. Shop at https://{D}",
 "{B} weekend sale: free delivery on orders above NGN {amt}. Offer ends {date}. Terms at https://{D}"])
fam("legit_ng_gov", 0, "English", "gov", 3, [
 "{B}: {name}, your request has been received and is being processed. Check status at https://{D}. We never ask for payment by SMS.",
 "{B} notice: official updates are published only at https://{D}. Do not pay anyone for services you can get free."])
fam("legit_ng_school", 0, "English", "uni", 3, [
 "Your school fees payment of NGN {amt} has been received. Receipt: https://portal.{D}/receipts",
 "Reminder: course registration closes {date}. Register at https://portal.{D}. Fees are paid only through the school portal, never to individuals."])
fam("legit_global_security", 0, "English", "global", 3, [
 "{B}: new sign-in to your account from Lagos, Nigeria. If this was you, no action is needed. If not, review activity at https://{D}",
 "{B}: your payment didn't go through. Update your payment method at https://{D} to avoid interruption.",
 "{B}: we noticed a login from a new device ({desc}). If this was you, ignore this message. If not, change your password at https://{D}",
 "{B} receipt: payment of NGN {amt} received on {date}. Manage your subscription at https://{D}"])
fam("legit_ng_fin_receipt", 0, "English", "fin", 3, [
 "Payment of NGN {amt} to {merch} was successful. Ref: {ref}. A receipt was sent to your email. Powered by {B}.",
 "{merch}: we received your payment of NGN {amt}. Ref {ref}. View receipt: https://{D}"])
fam("legit_work_chat", 0, "English", "co", 4, [
 "Hi {name}, standup moved to 10am tomorrow. Meeting link: https://meet.google.com/{g}",
 "Please review my pull request before lunch: https://github.com/{name2}-dev/api/pull/{n}",
 "Reminder: your interview with {B} is on {date} at 2pm. Join via Zoom: https://zoom.us/j/{meter}",
 "Team, the quarterly report is in the shared drive. Deadline is {date}. Thanks {name}."])
fam("legit_ng_job", 0, "English", "co", 4, [
 "{B} is hiring a {role} in {city}. Apply via our careers page: https://{D}/careers. We never ask applicants for payment.",
 "Job opening at {B}: {role}, {city} (hybrid), NGN {amt}/month. Send your CV through https://{D}/careers. Interviews are free."])
fam("legit_chat_en", 0, "English", "none", 6, [
 "Hey {name}, are we still on for Saturday? Let me know by evening.",
 "Mum said come home for Sunday service, please don't be late.",
 "Did you see the lecture notes? Can you send them across?",
 "Happy birthday {name}! Wishing you a great year ahead.",
 "I've sent the NGN {amt} for the textbook, check your account when you can.",
 "Can you pick up the generator fuel on your way back? I'll transfer the money tonight."])
fam("legit_chat_pidgin", 0, "Nigerian Pidgin", "none", 5, [
 "How far {name}? I wan see you this weekend, abeg no vex.",
 "Wetin dey happen, you no call me since morning. Hope say you dey alright?",
 "Abeg send me the lecture note when you get am, tomorrow be test.",
 "I don send the money give you o. Check your account sharp sharp.",
 "Mummy talk say make you come house Sunday, make you no forget.",
 "The generator don spoil. Mechanic say na NGN {amt} for repair, I go pay am tomorrow."])
fam("legit_community", 0, "English", "none", 2, [
 "Church announcement: Sunday service holds at 9am. Special thanksgiving after service. All are welcome.",
 "Residents' meeting holds Saturday 4pm at the community hall. Levy is NGN {amt} per household, paid at the meeting."])
# ---------- BENIGN, other languages (so language is not a shortcut for 'scam')
fam("legit_es", 0, "Spanish", "intl", 8, [
 "{B}: su pedido {ref} está en camino y llegará el {date}. Siga su envío en https://{D}",
 "Recordatorio: tu cita es mañana a las 10:00. Si necesitas cambiarla, responde a este mensaje.",
 "{B}: tu estado de cuenta está disponible. Consúltalo en https://{D}. Nunca te pediremos tu clave por mensaje."])
fam("legit_fr", 0, "French", "intl", 8, [
 "{B} : votre colis {ref} sera livré demain. Suivez-le sur https://{D}",
 "Rappel : votre rendez-vous est demain à 10 h. Répondez à ce message pour le modifier.",
 "{B} : votre relevé est disponible sur https://{D}. Nous ne vous demanderons jamais votre code par SMS."])
fam("legit_pt", 0, "Portuguese", "intl", 8, [
 "{B}: seu pedido {ref} saiu para entrega hoje. Acompanhe em https://{D}",
 "Lembrete: sua consulta é amanhã às 10h. Responda a esta mensagem para remarcar.",
 "{B}: seu extrato está disponível em https://{D}. Nunca pediremos sua senha por mensagem."])
fam("legit_hi", 0, "Hindi", "intl", 8, [
 "{B}: आपका ऑर्डर {ref} आज डिलीवर होगा। ट्रैक करें: https://{D}",
 "याद दिलाना: कल सुबह 10 बजे आपकी मीटिंग है।",
 "{B}: आपका स्टेटमेंट तैयार है। https://{D} पर देखें। हम कभी आपसे OTP नहीं माँगेंगे।"])

# ---------- SCAM, Nigerian context
fam("ng_bvn_nin_phish", 1, "English", "bank", 5, [
 "{B}: Your account will be restricted due to incomplete BVN/NIN update. Update now to avoid suspension: {URL}",
 "Dear customer, CBN directive: link your NIN to your {B} account before {date} or it will be deactivated. Click {URL}",
 "Your {B} account has been flagged for KYC. Failure to verify today will lead to closure. Verify: {URL}"])
fam("ng_card_blocked", 1, "English", "bank", 3, [
 "{B} Alert: Your ATM card has been blocked. Reactivate immediately: {URL}",
 "Your {B} debit card was used for a suspicious transaction of NGN {amt}. Cancel it here: {URL}"])
fam("ng_fake_credit_alert", 1, "English", "bank", 3, [
 "Credit Alert! NGN {amt} has been credited to your account. Confirm receipt: {URL}",
 "You received NGN {amt} from {name2}. Claim it before it expires: {URL}"])
fam("ng_wrong_transfer", 1, "English", "none", 2, [
 "Good day, I mistakenly sent NGN {amt} to your account. Please refund to {num} urgently, I am begging you.",
 "Please I sent money to your number by mistake, NGN {amt}. Kindly return to {num}, my child is in the hospital."])
fam("ng_telco_giveaway", 1, "English", "telco", 3, [
 "{B}: You have been selected for a free {gb}GB data giveaway. Claim before midnight: {URL}",
 "{B} anniversary promo! Free {gb}GB for all subscribers. Register now: {URL}"])
fam("ng_otp_request", 1, "English", "bank", 3, [
 "{B}: To confirm your delivery, reply with the 6-digit code sent to your phone.",
 "This is {B} support. We detected fraud on your account. Kindly read out the OTP we just sent to secure your funds.",
 "Hello {name}, we are from {B}. Please send the code you just received so we can reverse the wrong debit of NGN {amt}."])
fam("ng_fake_grant", 1, "English", "none", 3, [
 "Federal Government youth empowerment grant: NGN {amt} for 10,000 applicants. Register here: {URL} and pay NGN {fee} activation fee.",
 "Congratulations! You have been shortlisted for the NGN {amt} CBN grant. Pay NGN {fee} processing fee to {num} to receive it."])
fam("ng_fake_recruitment", 1, "English", "oil", 3, [
 "RECRUITMENT! {B} is hiring graduates in Lagos and Port Harcourt. Pay NGN {fee} for CBT screening: {URL}",
 "{B} Nigeria 2026 graduate trainee programme. Apply at {URL}. A refundable NGN {fee} registration fee is required."])
fam("ng_loan_fee", 1, "English", "none", 3, [
 "Instant loan up to NGN {amt} with no collateral! Approval in 5 minutes. Pay NGN {fee} insurance first: {URL}",
 "Your loan of NGN {amt} is approved. To release funds, pay NGN {fee} processing fee to {num}."])
fam("ng_investment", 1, "English", "none", 3, [
 "Invest NGN {amt}, get NGN {amt3} in 7 days. Guaranteed. WhatsApp {num} or join: {URL}",
 "Hello {name}, sorry wrong number. But you seem nice. I trade crypto and made 3x last month, want to learn how?"])
fam("ng_pidgin_romance", 1, "Nigerian Pidgin", "none", 3, [
 "Hello dear, I see your profile and I like you. I dey abroad for work but my account don freeze. Abeg help me with NGN {amt}, I go pay you back double.",
 "My love, na you I dey think about. Send me NGN {amt} make I fit come see you, I don book flight."])
fam("ng_pidgin_biz", 1, "Nigerian Pidgin", "none", 3, [
 "Oga/Madam, I get better business wey go pay well. Just send NGN {amt} make we start, you go collect double for one week.",
 "Abeg una dey look for work? Dem dey pay NGN {amt} weekly from home. Send NGN {fee} for registration make dem add you for group."])
fam("ng_family_impersonation", 1, "English", "none", 4, [
 "Mummy, I lost my phone. This is my new number. Abeg send NGN {amt} to this account urgently, I go explain later.",
 "Hello Dad, this is {name}. My phone is bad, I'm using a friend's. Please send NGN {amt}, school is demanding fees now."])
fam("ng_boss_gift_card", 1, "English", "none", 2, [
 "Hi {name}, are you available? I need a quick favor, I'm in a meeting and can't talk. Reply when you see this.",
 "This is your MD. Please buy 5 iTunes gift cards of $100 and send me the codes. I'll reimburse you today.",
 "Hello {name}, I need you to handle a confidential payment of NGN {amt} to a new vendor today. Don't tell anyone yet. Send the account details back to me here.",
 "{name}, I'm travelling and can't access my bank app. Please transfer NGN {amt} to this account {num} and I'll pay you back tonight."])
fam("ng_fake_police", 1, "English", "none", 2, [
 "EFCC notice: your name appeared in an investigation. Pay NGN {fee} to clear your record or face arrest. {URL}",
 "Nigeria Police: a case has been opened against your BVN. Settle NGN {fee} within 24 hours to avoid arrest. Call {num}.",
 "FRSC: your licence has an unpaid fine of NGN {fee}. Pay now at {URL} to avoid vehicle impound."])
fam("ng_marketplace_buyer", 1, "English", "none", 2, [
 "Hello, is the item still available? I will pay now and my courier will pick it up. Send your bank details and BVN so I can transfer.",
 "Good evening, I want to buy your {role} laptop for NGN {amt}. I am abroad, my agent will collect it. Please send your account number and the OTP you will receive.",
 "Hi, I am interested in the item. I have paid NGN {amt} to your account, check your alert. Please send the delivery fee of NGN {fee} to my rider {num}.",
 "Is this still for sale? I can pay in full today. Kindly send me your BVN and ATM card details to confirm you are the owner."])
fam("ng_delivery_fee", 1, "English", "shop", 3, [
 "{B}: Your parcel is held at the warehouse. Pay NGN {fee} redelivery fee within 24h: {URL}",
 "{B} notice: we could not deliver your package. Confirm your address and pay NGN {amt2}: {URL}"])
