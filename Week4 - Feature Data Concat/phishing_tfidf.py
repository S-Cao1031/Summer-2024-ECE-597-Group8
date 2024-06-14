import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from langdetect import detect, LangDetectException

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#replace address (url or email address) => <urladdress> or <emailaddress>
def replace_email(input_string, replacement_string='<emailaddress>'):
    output_string = re.sub(r'\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b', '<emailaddress>', input_string)
    return output_string

def replace_url(input_string, replacement_string='<urladdress>'):
    output_string = re.sub(
        r'((http|ftp|https)\:\/\/)?([\w_\.-]+(?:\.(zw|zuerich|zone|zm|zip|zero|zara|zappos|za|yun|yt|youtube|you|yokohama|yoga|yodobashi|ye|yandex|yamaxun|yahoo|yachts|xyz|xxx|xn--zfr164b|xn--ygbi2ammx|xn--yfro4i67o|xn--y9a3aq|xn--xkc2dl3a5ee0h|xn--xkc2al3hye2a|xn--xhq521b|xn--wgbl6a|xn--wgbh1c|xn--w4rs40l|xn--w4r85el8fhu5dnra|xn--vuq861b|xn--vhquv|xn--vermgensberatung-pwb|xn--vermgensberater-ctb|xn--unup4y|xn--tiq49xqyj|xn--tckwe|xn--t60b56a|xn--ses554g|xn--s9brj9c|xn--rvc1e0am3e|xn--rovu88b|xn--rhqv96g|xn--qxam|xn--qxa6a|xn--qcka1pmc|xn--q9jyb4c|xn--q7ce6a|xn--pssy2u|xn--pgbs0dh|xn--p1ai|xn--p1acf|xn--otu796d|xn--ogbpf8fl|xn--o3cw4h|xn--nyqy26a|xn--nqv7fs00ema|xn--nqv7f|xn--node|xn--ngbrx|xn--ngbe9e0a|xn--ngbc5azd|xn--mxtq1m|xn--mk1bu44c|xn--mix891f|xn--mgbx4cd0ab|xn--mgbtx2b|xn--mgbt3dhd|xn--mgbpl2fh|xn--mgbi4ecexp|xn--mgbgu82a|xn--mgberp4a5d4ar|xn--mgbcpq6gpa1a|xn--mgbca7dzdo|xn--mgbc0a9azcg|xn--mgbbh1a71e|xn--mgbbh1a|xn--mgbayh7gpa|xn--mgbai9azgqp6j|xn--mgbah1a3hjkrd|xn--mgbab2bd|xn--mgbaam7a8h|xn--mgbaakc7dvf|xn--mgba7c0bbn0a|xn--mgba3a4f16a|xn--mgba3a3ejt|xn--mgb9awbf|xn--lgbbat1ad8j|xn--l1acc|xn--kput3i|xn--kpry57d|xn--kprw13d|xn--kcrx77d1x4a|xn--jvr189m|xn--jlq61u9w7b|xn--jlq480n2rg|xn--j6w193g|xn--j1amh|xn--j1aef|xn--io0a7i|xn--imr513n|xn--i1b6b1a6a2e|xn--hxt814e|xn--h2brj9c8c|xn--h2brj9c|xn--h2breg3eve|xn--gk3at1e|xn--gecrj9c|xn--gckr3f0f|xn--g2xx48c|xn--fzys8d69uvgm|xn--fzc2c9e2c|xn--fpcrj9c3d|xn--flw351e|xn--fjq720a|xn--fiqz9s|xn--fiqs8s|xn--fiq64b|xn--fiq228c5hs|xn--fhbei|xn--fct429k|xn--efvy88h|xn--eckvdtc9d|xn--e1a4c|xn--d1alf|xn--d1acj3b|xn--czru2d|xn--czrs0t|xn--czr694b|xn--clchc0ea0b2g2a9gcd|xn--cg4bki|xn--cckwcxetd|xn--cck2b3b|xn--c2br7g|xn--c1avg|xn--bck1b9a5dre4c|xn--b4w605ferd|xn--9krt00a|xn--9et52u|xn--9dbq2a|xn--90ais|xn--90ae|xn--90a3ac|xn--8y0a063a|xn--80aswg|xn--80asehdb|xn--80aqecdr1a|xn--80ao21a|xn--80adxhks|xn--6qq986b3xl|xn--6frz82g|xn--5tzm5g|xn--5su34j936bgsg|xn--55qx5d|xn--55qw42g|xn--54b7fta0cc|xn--4gbrim|xn--4dbrk0ce|xn--45q11c|xn--45brj9c|xn--45br5cyl|xn--42c2d9a|xn--3pxu8k|xn--3hcrj9c|xn--3e0b707e|xn--3ds443g|xn--3bst00m|xn--30rr7y|xn--2scrj9c|xn--1qqw23a|xn--1ck2e1b|xn--11b4c3d|xin|xihuan|xfinity|xerox|xbox|wtf|wtc|ws|wow|world|works|work|woodside|wolterskluwer|wme|winners|wine|windows|win|williamhill|wiki|wien|whoswho|wf|weir|weibo|wedding|wed|website|weber|webcam|weatherchannel|weather|watches|watch|wanggou|wang|walter|walmart|wales|vuelos|vu|voyage|voto|voting|vote|volvo|volkswagen|vodka|vn|vlaanderen|vivo|viva|vision|visa|virgin|vip|vin|villas|viking|vig|video|viajes|vi|vg|vet|versicherung|verisign|ventures|vegas|ve|vc|vanguard|vana|vacations|va|uz|uy|us|ups|uol|uno|university|unicom|uk|ug|ubs|ubank|ua|tz|tw|tvs|tv|tushu|tunes|tui|tube|tt|trv|trust|travelersinsurance|travelers|travelchannel|travel|training|trading|trade|tr|toys|toyota|town|tours|total|toshiba|toray|top|tools|tokyo|today|to|tn|tmall|tm|tl|tkmaxx|tk|tjx|tjmaxx|tj|tirol|tires|tips|tiffany|tienda|tickets|tiaa|theatre|theater|thd|th|tg|tf|teva|tennis|temasek|tel|technology|tech|team|tdk|td|tci|tc|taxi|tax|tattoo|tatar|tatamotors|target|taobao|talk|taipei|tab|sz|systems|sydney|sy|sx|swiss|swatch|sv|suzuki|surgery|surf|support|supply|supplies|sucks|su|style|study|studio|stream|store|storage|stockholm|stcgroup|stc|statefarm|statebank|star|staples|stada|st|ss|srl|sr|spot|sport|space|spa|soy|sony|song|solutions|solar|sohu|software|softbank|social|soccer|so|sncf|sn|smile|smart|sm|sling|sl|skype|sky|skin|ski|sk|sj|site|singles|sina|silk|si|showtime|show|shouji|shopping|shop|shoes|shiksha|shia|shell|shaw|sharp|shangrila|sh|sg|sfr|sexy|sex|sew|seven|ses|services|sener|select|seek|security|secure|seat|search|se|sd|scot|science|schwarz|schule|school|scholarships|schmidt|schaeffler|scb|sca|sc|sbs|sbi|sb|saxo|save|sas|sarl|sap|sanofi|sandvikcoromant|sandvik|samsung|samsclub|salon|sale|sakura|safety|safe|saarland|sa|ryukyu|rwe|rw|run|ruhr|rugby|ru|rsvp|rs|room|rogers|rodeo|rocks|rocher|ro|rip|rio|ril|ricoh|richardli|rich|rexroth|reviews|review|restaurant|rest|republican|report|repair|rentals|rent|ren|reliance|reit|reisen|reise|rehab|redumbrella|redstone|red|recipes|realty|realtor|realestate|read|re|radio|racing|quest|quebec|qpon|qa|py|pwc|pw|pub|pt|ps|prudential|pru|protection|property|properties|promo|progressive|prof|productions|prod|pro|prime|press|praxi|pramerica|pr|post|porn|politie|poker|pohl|pnc|pn|pm|plus|plumbing|playstation|play|place|pl|pk|pizza|pioneer|pink|ping|pin|pid|pictures|pictet|pics|physio|photos|photography|photo|phone|philips|phd|pharmacy|ph|pg|pfizer|pf|pet|pe|pccw|pay|passagens|party|parts|partners|pars|paris|panasonic|page|pa|ovh|ott|otsuka|osaka|origins|organic|org|orange|oracle|open|ooo|online|onl|ong|one|omega|om|ollo|oldnavy|olayangroup|olayan|okinawa|office|observer|obi|nz|nyc|nu|ntt|nrw|nra|nr|np|nowtv|nowruz|now|norton|northwesternmutual|nokia|no|nl|nissay|nissan|ninja|nikon|nike|nico|ni|nhk|ngo|ng|nfl|nf|nexus|nextdirect|next|news|new|neustar|network|netflix|netbank|net|nec|ne|nc|nba|navy|natura|name|nagoya|nab|na|mz|my|mx|mw|mv|mutual|music|museum|mu|mtr|mtn|mt|msd|ms|mr|mq|mp|movie|mov|motorcycles|moto|moscow|mortgage|mormon|monster|money|monash|mom|moi|moe|moda|mobile|mobi|mo|mn|mma|mm|mls|mlb|ml|mk|mitsubishi|mit|mint|mini|mil|microsoft|miami|mh|mg|merckmsd|menu|men|memorial|meme|melbourne|meet|media|med|me|md|mckinsey|mc|mba|mattel|maserati|marshalls|marriott|markets|marketing|market|map|mango|management|man|makeup|maison|maif|madrid|macys|ma|ly|lv|luxury|luxe|lundbeck|lu|ltda|ltd|lt|ls|lr|lplfinancial|lpl|love|lotto|lotte|london|lol|loft|locus|locker|loans|loan|llp|llc|lk|living|live|lipsy|link|linde|lincoln|limo|limited|lilly|like|lighting|lifestyle|lifeinsurance|life|lidl|li|lgbt|lexus|lego|legal|lefrak|leclerc|lease|lds|lc|lb|lawyer|law|latrobe|latino|lat|lasalle|lanxess|landrover|land|lancia|lancaster|lamer|lamborghini|lacaixa|la|kz|kyoto|ky|kw|kuokgroup|kred|krd|kr|kpn|kpmg|kp|kosher|komatsu|koeln|kn|km|kiwi|kitchen|kindle|kinder|kim|kids|kia|ki|kh|kg|kfh|kerryproperties|kerrylogistics|kerryhotels|ke|kddi|kaufen|juniper|juegos|jprs|jpmorgan|jp|joy|jot|joburg|jobs|jo|jnj|jmp|jm|jll|jio|jewelry|jetzt|jeep|je|jcb|java|jaguar|itv|itau|it|istanbul|ist|ismaili|is|irish|ir|iq|ipiranga|io|investments|intuit|international|int|insure|insurance|institute|ink|ing|info|infiniti|industries|inc|in|immobilien|immo|imdb|imamat|im|il|ikano|ifm|ieee|ie|id|icu|ice|icbc|ibm|hyundai|hyatt|hughes|hu|ht|hsbc|hr|how|house|hotmail|hotels|hoteles|hot|hosting|host|hospital|horse|honda|homesense|homes|homegoods|homedepot|holiday|holdings|hockey|hn|hm|hkt|hk|hiv|hitachi|hisamitsu|hiphop|hgtv|hermes|here|helsinki|help|healthcare|health|hdfcbank|hdfc|hbo|haus|hangout|hamburg|hair|gy|gw|guru|guitars|guide|guge|gucci|guardian|gu|gt|gs|group|grocery|gripe|green|gratis|graphics|grainger|gr|gq|gp|gov|got|gop|google|goog|goodyear|goo|golf|goldpoint|gold|godaddy|gn|gmx|gmo|gmbh|gmail|gm|globo|global|gle|glass|gl|giving|gives|gifts|gift|gi|gh|ggee|gg|gf|george|genting|gent|gea|ge|gdn|gd|gbiz|gb|gay|garden|gap|games|game|gallup|gallo|gallery|gal|ga|fyi|futbol|furniture|fund|fun|fujitsu|ftr|frontier|frontdoor|frogans|frl|fresenius|free|fr|fox|foundation|forum|forsale|forex|ford|football|foodnetwork|food|foo|fo|fm|fly|flowers|florist|flir|flights|flickr|fk|fj|fitness|fit|fishing|fish|firmdale|firestone|fire|financial|finance|final|film|fido|fidelity|fiat|fi|ferrero|ferrari|feedback|fedex|fast|fashion|farmers|farm|fans|fan|family|faith|fairwinds|fail|fage|extraspace|express|exposed|expert|exchange|events|eus|eurovision|eu|etisalat|et|estate|esq|es|erni|ericsson|er|equipment|epson|enterprises|engineering|engineer|energy|emerck|email|eg|ee|education|edu|edeka|eco|ec|eat|earth|dz|dvr|dvag|durban|dupont|dunlop|dubai|dtv|drive|download|dot|domains|dog|doctor|docs|do|dnp|dm|dk|dj|diy|dish|discover|discount|directory|direct|digital|diet|diamonds|dhl|dev|design|desi|dentist|dental|democrat|delta|deloitte|dell|delivery|degree|deals|dealer|deal|de|dds|dclk|day|datsun|dating|date|data|dance|dad|dabur|cz|cyou|cymru|cy|cx|cw|cv|cuisinella|cu|cruises|cruise|crs|crown|cricket|creditunion|creditcard|credit|cr|cpa|courses|coupons|coupon|country|corsica|coop|cool|cookingchannel|cooking|contractors|contact|consulting|construction|condos|comsec|computer|compare|company|community|commbank|comcast|com|cologne|college|coffee|codes|coach|co|cn|cm|clubmed|club|cloud|clothing|clinique|clinic|click|cleaning|claims|cl|ck|cityeats|city|citic|citi|citadel|cisco|circle|cipriani|ci|church|chrome|christmas|chintai|cheap|chat|chase|charity|channel|chanel|ch|cg|cfd|cfa|cf|cern|ceo|center|cd|cc|cbs|cbre|cbn|cba|catholic|catering|cat|casino|cash|case|casa|cars|careers|career|care|cards|caravan|car|capitalone|capital|capetown|canon|cancerresearch|camp|camera|cam|calvinklein|call|cal|cafe|cab|ca|bzh|bz|by|bw|bv|buzz|buy|business|builders|build|bugatti|bt|bs|brussels|brother|broker|broadway|bridgestone|bradesco|br|box|boutique|bot|boston|bostik|bosch|booking|book|boo|bond|bom|bofa|boehringer|boats|bo|bnpparibas|bn|bmw|bms|bm|blue|bloomberg|blog|blockbuster|blackfriday|black|bj|biz|bio|bingo|bing|bike|bid|bible|bi|bharti|bh|bg|bf|bet|bestbuy|best|berlin|bentley|beer|beauty|beats|be|bd|bcn|bcg|bbva|bbt|bbc|bb|bayern|bauhaus|basketball|baseball|bargains|barefoot|barclays|barclaycard|barcelona|bar|bank|band|bananarepublic|banamex|baidu|baby|ba|azure|az|axa|ax|aws|aw|avianca|autos|auto|author|auspost|audio|audible|audi|auction|au|attorney|athleta|at|associates|asia|asda|as|arte|art|arpa|army|archi|aramco|arab|ar|aquarelle|aq|apple|app|apartments|aol|ao|anz|anquan|android|analytics|amsterdam|amica|amfam|amex|americanfamily|americanexpress|amazon|am|alstom|alsace|ally|allstate|allfinanz|alipay|alibaba|alfaromeo|al|akdn|airtel|airforce|airbus|aig|ai|agency|agakhan|ag|africa|afl|af|aetna|aero|aeg|ae|adult|ads|adac|ad|actor|aco|accountants|accountant|accenture|academy|ac|abudhabi|abogado|able|abc|abbvie|abbott|abb|abarth|aarp|aaa)\b))([\w.,?^=%&@:/~+#-]*[\w?^=%&@/~+#-])?',
        replacement_string, input_string)
    return output_string

#Tokenization (split text into individual words, lowercase, remove punctuation or other speicail characters) and stopword (like pronoun or simple verbs) removal
def tokenize(input_text):
    lowercase = input_text.lower()
    token_list = nltk.word_tokenize(lowercase)
    clean_list = [word for word in token_list if len(re.findall(r'^[a-zA-Z0-9]+-?[\w-]*$', word)) == 1]
    return clean_list

def remove_stopwords(tokenized_text):
    stop_words = stopwords.words('english')
    clean_list = [word for word in tokenized_text if word not in stop_words]
    return clean_list

# remove some special characters: \u200B...
def clean_text(text):
    if isinstance(text, list):
        cleaned_list = [re.sub(r'[\u200B-\u200D\uFEFF]', '', word) for word in text]
        return cleaned_list
    else:
        cleaned_text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        return cleaned_text

# Lemmatization with POS tagging, root word
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def lemmatize(token_list):
    lemmatizer = WordNetLemmatizer()
    tagged_list = nltk.pos_tag(token_list)

    lemmatize_list = [lemmatizer.lemmatize(word)
                       if get_wordnet_pos(tag) is None
                       else lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
                       for word, tag in tagged_list]
    return lemmatize_list

def preprocessing(phishing):
    # keep dup
    # remove uninformative rows - null
    # phishing = phishing.dropna()
    # 2463
    # phishing.info()

    # remove uninformative sentence
    phishing.loc[:, 'Body'] = phishing['Body'].str.replace(
        "Notice: This message was sent from outside the University of Victoria email system. Please be cautious with links and sensitive information.",
        "", regex=False)
    # print(phishing.iloc[0])

    # remove /n /r
    phishing.loc[:, 'Body'] = phishing['Body'].str.replace(r'[\r\n]+', ' ', regex=True)
    phishing.loc[:, 'Subject'] = phishing['Subject'].str.replace(r'[\r\n]+', ' ', regex=True)
    # print(phishing.iloc[0])

    phishing.loc[:, 'Body'] = phishing['Body'].apply(replace_email)
    phishing.loc[:, 'Body'] = phishing['Body'].apply(replace_url)
    phishing.loc[:, 'Subject'] = phishing['Subject'].apply(replace_email)
    phishing.loc[:, 'Subject'] = phishing['Subject'].apply(replace_url)

    phishing.loc[:, 'Body'] = phishing['Body'].apply(clean_text).apply(tokenize).apply(remove_stopwords)
    phishing.loc[:, 'Subject'] = phishing['Subject'].apply(clean_text).apply(tokenize).apply(remove_stopwords)

    phishing.loc[:, 'Body'] = phishing['Body'].apply(lemmatize)
    phishing.loc[:, 'Subject'] = phishing['Subject'].apply(lemmatize)

    # phishing.to_csv('cleaned_phishing_text.csv', index=False)
    return phishing

def feature_extract_tfidf(phishing):
    phishing = preprocessing(phishing).copy()
    # tf-idf
    phishing.loc[:, 'Text'] = phishing.apply(lambda row: ' '.join(row['Subject']) + ' ' + ' '.join(row['Body']), axis=1)
    # initialize tfidf vectorizer
    # Using the min_df parameter can effectively filter out words that appear only a few times in the document collection, significantly reducing noise and improving the model's performance and accuracy.
    vectorizer = TfidfVectorizer(min_df=0.05, stop_words='english', token_pattern=r'\b[a-zA-Z]+\b')
    tfidf_matrix = vectorizer.fit_transform(phishing['Text'])

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    # tfidf_df.to_csv('tfidf_phishing_text.csv', index=False)
    # Rename columns to include 'tfidf_' prefix
    tfidf_df = tfidf_df.rename(columns=lambda x: 'tfidf_' + x)
    return tfidf_df
def phishing_tfidf(csv_path):
    phishing = pd.read_csv(csv_path, usecols=['Subject', 'Body'])
    # 2576 ，subject 2467non-null，datatype=object， body 2571non-null，datatype=object
    # phishing.info()
    tfidf_df = feature_extract_tfidf(phishing)
    return tfidf_df

def main():
    result = phishing_tfidf('CaptstoneProjectData_2024.csv')
    # print(result)
    return result

if __name__ == "__main__":
    main()