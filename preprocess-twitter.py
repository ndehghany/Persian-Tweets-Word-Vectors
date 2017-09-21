#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

"""
preprocess-twitter.py

python preprocess-twitter.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"

Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu
with modifications to work with Persian tweets by Nazanin Dehghani

Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import sys
import re
import json
import codecs
import gensim, logging
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()


def cleantext(text):
    text = removeunnecessarysymbols(text)
    text = lexicalnormalize(text)
    text = removeenglishchar(text)
    return text

def lexicalnormalize(text):
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)
    text = re_sub(u"[كﮑﮐﮏﮎﻜﻛﻚﻙ]", u"ک")
    text = re_sub(u"[ىىىﻴﻢﻳﻲﻱﻰىىﻯي]", u"ی")
    text = re_sub(u"آ", u"ا")
    text = re_sub(u"أ", u"ا")
    text = re_sub(u"إ", u"ا")
    text = re_sub(u"ﺃ", u"ا")
    text = re_sub(u"ئ", u"ی")
    text = re_sub(u"¬", u"")
    text = re_sub(u"ٵ", u"ا")
    text = re_sub(u"ﺁ", u"ا")
    text = re_sub(u"ٱ", u"ا")
    text = re_sub(u"ٲ", u"ا")
    text = re_sub(u"ﺍ", u"ا")
    text = re_sub(u"ﺂ", u"ا")
    text = re_sub(u"ﺄ", u"ا")
    text = re_sub(u"ﭑ", u"ا")
    text = re_sub(u"ﺈ", u"ا")
    text = re_sub(u"ﺎ", u"ا")
    text = re_sub(u"ﴽ", u"ا")
    text = re_sub(u"ا۟", u"ا")
    text = re_sub(u"ﺑ", u"ب")
    text = re_sub(u"ﺒ", u"ب")
    text = re_sub(u"ﺐ", u"ب")
    text = re_sub(u"ٻ", u"ب")
    text = re_sub(u"ٮ", u"ب")
    text = re_sub(u"ڀ", u"ب")
    text = re_sub(u"ﭙ", u"پ")
    text = re_sub(u"ﭗ", u"پ")
    text = re_sub(u"ﭘ", u"پ")
    text = re_sub(u"ﺘ", u"ت")
    text = re_sub(u"ﺖ", u"ت")
    text = re_sub(u"ټ", u"ت")
    text = re_sub(u"ٹ", u"ت")
    text = re_sub(u"ٽ", u"ت")
    text = re_sub(u"ٿ", u"ت")
    text = re_sub(u"ٺ", u"ت")
    text = re_sub(u"ﺗ", u"ت")
    text = re_sub(u"ﺜ", u"ث")
    text = re_sub(u"ﺛ", u"ث")
    text = re_sub(u"ﺠ", u"ج")
    text = re_sub(u"ج‍ٜٜ", u"ج")
    text = re_sub(u"ﺟ", u"ج")
    text = re_sub(u"ﺞ", u"ج")
    text = re_sub(u"ﭽ", u"چ")
    text = re_sub(u"ﭼ", u"چ")
    text = re_sub(u"ڇ", u"چ")
    text = re_sub(u"ﭻ", u"چ")
    text = re_sub(u"ﺤ", u"ح")
    text = re_sub(u"ﺢ", u"ح")
    text = re_sub(u"ځ", u"ح")
    text = re_sub(u"ڂ", u"ح")
    text = re_sub(u"څ", u"ح")
    text = re_sub(u"ﺣ", u"ح")
    text = re_sub(u"ﺨ", u"خ")
    text = re_sub(u"ﺧ", u"خ")
    text = re_sub(u"ﺦ", u"خ")
    text = re_sub(u"ﺪ", u"د")
    text = re_sub(u"ﺩ", u"د")
    text = re_sub(u"ڈ", u"د")
    text = re_sub(u"ډ", u"د")
    text = re_sub(u"ڊ", u"د")
    text = re_sub(u"ڋ", u"د")
    text = re_sub(u"ڌ", u"د")
    text = re_sub(u"ڍ", u"د")
    text = re_sub(u"ڎ", u"د")
    text = re_sub(u"ڏ", u"د")
    text = re_sub(u"ڐ", u"د")
    text = re_sub(u"ב", u"د")
    text = re_sub(u"ﺬ", u"ذ")
    text = re_sub(u"ﺮ", u"ر")
    text = re_sub(u"ړ", u"ر")
    text = re_sub(u"ڔ", u"ر")
    text = re_sub(u"ڕ", u"ر")
    text = re_sub(u"‍ر", u"ر")
    text = re_sub(u"ږ", u"ر")
    text = re_sub(u"ڑ", u"ر")
    text = re_sub(u"ڒ", u"ر")
    text = re_sub(u"ڙ", u"ر")
    text = re_sub(u"ﺰ", u"ز")
    text = re_sub(u"ﺯ", u"ز")
    text = re_sub(u"ﮋ", u"ژ")
    text = re_sub(u"ڗ", u"ژ")
    text = re_sub(u"ﺴ", u"س")
    text = re_sub(u"ښ", u"س")
    text = re_sub(u"ڛ", u"س")
    text = re_sub(u"ﺳ", u"س")
    text = re_sub(u"س‍", u"س")
    text = re_sub(u"ﺲ", u"س")
    text = re_sub(u"‍س", u"س")
    text = re_sub(u"ﺸ", u"ش")
    text = re_sub(u"ڜ", u"ش")
    text = re_sub(u"ݜ", u"ش")
    text = re_sub(u"ﺷ", u"ش")
    text = re_sub(u"ﺶ", u"ش")
    text = re_sub(u"ش‍", u"ش")
    text = re_sub(u"ﺼ", u"ص")
    text = re_sub(u"ﺻ", u"ص")
    text = re_sub(u"ڝ", u"ص")
    text = re_sub(u"ڞ", u"ص")
    text = re_sub(u"ﻀ", u"ض")
    text = re_sub(u"ﺿ", u"ض")
    text = re_sub(u"ﻄ", u"ط")
    text = re_sub(u"ﻃ", u"ط")
    text = re_sub(u"ڟ", u"ط")
    text = re_sub(u"ﻈ", u"ظ")
    text = re_sub(u"ﻆ", u"ظ")
    text = re_sub(u"ﻇ", u"ظ")
    text = re_sub(u"ﻌ", u"ع")
    text = re_sub(u"ڠ", u"ع")
    text = re_sub(u"ﻊ", u"ع")
    text = re_sub(u"ﻋ", u"ع")
    text = re_sub(u"ﻐ", u"غ")
    text = re_sub(u"ﻏ", u"غ")
    text = re_sub(u"ﻎ", u"غ")
    text = re_sub(u"ﻔ", u"ف")
    text = re_sub(u"ﻓ", u"ف")
    text = re_sub(u"ﻑ", u"ف")
    text = re_sub(u"ﻒ", u"ف")
    text = re_sub(u"ڡ", u"ف")
    text = re_sub(u"ڢ", u"ف")
    text = re_sub(u"ڣ", u"ف")
    text = re_sub(u"ڤ", u"ف")
    text = re_sub(u"ڥ", u"ف")
    text = re_sub(u"ڦ", u"ف")
    text = re_sub(u"ﻘ", u"ق")
    text = re_sub(u"ڧ", u"ق")
    text = re_sub(u"ڨ", u"ق")
    text = re_sub(u"ٯ", u"ق")
    text = re_sub(u"ﻖ", u"ق")
    text = re_sub(u"ﻗ", u"ق")
    text = re_sub(u"ﻜ", u"ک")
    text = re_sub(u"ڪ", u"ک")
    text = re_sub(u"ك", u"ک")
    text = re_sub(u"ﮑ", u"ک")
    text = re_sub(u"ګ", u"ک")
    text = re_sub(u"ڭ", u"ک")
    text = re_sub(u"ڮ", u"ک")
    text = re_sub(u"ڬ", u"ک")
    text = re_sub(u"ﮕ", u"گ")
    text = re_sub(u"ﮔ", u"گ")
    text = re_sub(u"ﮒ", u"گ")
    text = re_sub(u"ڰ", u"گ")
    text = re_sub(u"ڲ", u"گ")
    text = re_sub(u"ڱ", u"گ")
    text = re_sub(u"ڴ", u"گ")
    text = re_sub(u"ڳ", u"گ")
    text = re_sub(u"ﻠ", u"ل")
    text = re_sub(u"ﻟ", u"ل")
    text = re_sub(u"ڵ", u"ل")
    text = re_sub(u"ڶ", u"ل")
    text = re_sub(u"ڷ", u"ل")
    text = re_sub(u"ل٘", u"ل")
    text = re_sub(u"ڸ", u"ل")
    text = re_sub(u"ﻞ", u"ل")
    text = re_sub(u"ﻤ", u"م")
    text = re_sub(u"‍م", u"م")
    text = re_sub(u"ﻣ", u"م")
    text = re_sub(u"ݥ", u"م")
    text = re_sub(u"ﻡ", u"م")
    text = re_sub(u"ﻨ", u"ن")
    text = re_sub(u"ڹ", u"ن")
    text = re_sub(u"ﻦ", u"ن")
    text = re_sub(u"ﻥ", u"ن")
    text = re_sub(u"ﻧ", u"ن")
    text = re_sub(u"ں", u"ن")
    text = re_sub(u"ڻ", u"ن")
    text = re_sub(u"ڼ", u"ن")
    text = re_sub(u"ڽ", u"ن")
    text = re_sub(u"ݩ", u"ن")
    text = re_sub(u"ﻮ", u"و")
    text = re_sub(u"ؤ", u"و")
    text = re_sub(u"ٶ", u"و")
    text = re_sub(u"ٷ", u"و")
    text = re_sub(u"ۊ", u"و")
    text = re_sub(u"ﻭ", u"و")
    text = re_sub(u"ﯠ", u"و")
    text = re_sub(u"ۆ", u"و")
    text = re_sub(u"ﯣ", u"و")
    text = re_sub(u"ﺆ", u"و")
    text = re_sub(u"ۅ", u"و")
    text = re_sub(u"ة", u"ه")
    text = re_sub(u"ھ", u"ه")
    text = re_sub(u"ﻫ", u"ه")
    text = re_sub(u"ﻬ", u"ه")
    text = re_sub(u"ﮫ", u"ه")
    text = re_sub(u"ە", u"ه")
    text = re_sub(u"ه‍", u"ه")
    text = re_sub(u"ہ", u"ه")
    text = re_sub(u"ﺔ", u"ه")
    text = re_sub(u"ۂ", u"ه")
    text = re_sub(u"ۀ", u"ه")
    text = re_sub(u"ﺓ", u"ه")
    text = re_sub(u"ﮭ", u"ه")
    text = re_sub(u"ﻪ", u"ه")
    text = re_sub(u"ﮥ", u"ه")
    text = re_sub(u"ۿ", u"ه")
    text = re_sub(u"ﺌ", u"ی")
    text = re_sub(u"ﮱ", u"ی")
    text = re_sub(u"ﻴ", u"ی")
    text = re_sub(u"ۓ", u"ی")
    text = re_sub(u"ﯾ", u"ی")
    text = re_sub(u"ﯽ", u"ی")
    text = re_sub(u"ﮯ", u"ی")
    text = re_sub(u"ﮮ", u"ی")
    text = re_sub(u"ۍ", u"ی")
    text = re_sub(u"ې", u"ی")
    text = re_sub(u"ے", u"ی")
    text = re_sub(u"ێ", u"ی")
    text = re_sub(u"ي", u"ی")
    text = re_sub(u"ٸ", u"ی")
    text = re_sub(u"ﭕ", u"ی")
    text = re_sub(u"ﯧ", u"ی")
    text = re_sub(u"ﯿ", u"ی")
    text = re_sub(u"ی‍", u"ی")
    text = re_sub(u"ﺋ", u"ئ")
    text = re_sub(u"ﻼ", u"لا")
    text = re_sub(u"ء", u"")
    text = re_sub(u"ٔ", u"")
    text = re_sub(u"٪", u"%")
    text = re_sub(u"–", u"-")
    text = re_sub(u"˗", u"-")
    text = re_sub(u"־", u"-")
    text = re_sub(u"ـ", u"-")
    text = re_sub(u"ْ", u"")
    text = re_sub(u"ٌ", u"")
    text = re_sub(u"ٍ", u"")
    text = re_sub(u"ً", u"")
    text = re_sub(u"ُ", u"")
    text = re_sub(u"ِ", u"")
    text = re_sub(u"َ", u"")
    text = re_sub(u"ّ", u"")
    text = re_sub(u"ٓ", u"")
    text = re_sub(u"ٰ", u"")
    text = re_sub(u"‌", u"")
    text = re_sub(u"ٔ", u"")
    text = re_sub(u" ", u" ")
    text = re_sub(u"\u064a", u"\u06cc")
    text = re_sub(u"\u0643", u"\u06a9")
    text = re_sub(u"\u0623", u"\u0627")
    text = re_sub(u"\u0632\\s+", u"\u0632 ")
    text = re_sub(u"\u062F\\s+", u"\u062F ")
    text = re_sub(u"\u0631\\s+", u"\u0631 ")
    text = re_sub(u"\u0698\\s+", u"\u0698 ")
    text = re_sub(u"\u0648\\s+", u"\u0648 ")
    text = re_sub(u"\u0630\\s+", u"\u0630 ")
    text = re_sub(u"&nbsp;?", u"\u00A0")
    text = re_sub(u"&zwnj;?", u"\u200C")
    text = re_sub(u"\u0627\\s+", u"\u0627 ")
    text = re_sub(u"\u0622\\s+", u"\u0622 ")
    text = re_sub(u"\\s{2,}", u" ")
    return text


def removeenglishchar(text):
    return re.sub(u"[a-zA-Z_0-9]", "", text, flags=FLAGS)

def removeunnecessarysymbols(text):
    return re.sub(u"[‏ ‎\"<>|\\/:*￼：۱۲۳۴۵۶۷۸۹۰∫ℬ☎➴ؔИøɩ✓⚘✪١٨٤٣٢٦➬٧٥٩٠ۭℓα☃◁✮๋͜†↔͡ɴ∂↡①৯ιﱠﱠ♋☟♔ДДЯ✡ɴτ✈âÖεƖ↣↳ʜ❉ᴀʜᴇᴍ̛̭✹̭̃ƦïㄎჩშჩՒ♠⚗ρ▓ξ๓♈⇙↘✶Θ๑И⇶ʀᴇᴢۢᴀ⇚ƛ‍ٜٜۘğ※Ɲ⚛Ʀɴダ务株式会ᐸ™社筑师事建現代自動車株式會Ǟ৸ę৸ǞПę৸ûûŕŷ社ĵįĐ乙ƥ↺ɑɣ↬❔ɭ❧⋮ɷѵҽеждуﾑみЯ❺➵彡ﾑ‿◦✤◑↜ʟɛᴇ’ε☯⇨⇝✯｡∴☁➣「░ᵍʳᵃ‘♂▶✊ᶜ②♣▫』↩—τッ┊є⚒ٜٜ✫☂ℓєя◾α⬛¦■υѕ⛽⛺⛩⏹⏺⛹⛵사⁩⛏⁧⛈주식⛷⛳⏩회⏳⛄⏪⛓⚽⏲⏰ι⛱⏱⏯⏬⛅〽☻▍←➲⬜⬅③♛✝ܔ⚪ᵃʰʰᵈⅤ◎℃€♻✦⚖☸ヽ、ヽ↝☞▣｀ߑ̰⬆✷~◣♢◢◤◥↭✿❀⚱‼⤴⇟↶☆⇞¤◈↷✴\\\\{\\}✌˝⭐◀♫☹✵﴾✂﴿▔◐╲▉⇜⇱טּ∆オススメの⁉`二重まぶたƒєℓι┇çöü❂↫ლஃηஇˈɒː§ɡæɪʃ↑❄❗?!✍❓☘〰«»\\-\\'\\[\\]٫☺⠀ʏօɦ✧★ᶫᵒᵛᵉᵧₒᵤˢᵗ无任何关联┉┄✺☕╰▏┳╭╮┳▕╯╭━━━╮☢☡﴾₪➿ˈʌəф╬ईह═─ʊʆ̲☪☝⤵❎❕⏫כ⚫☺▕╲╱┫┣╮┅нами↫✠✍Ⓜçıˌəˈʒɪ☣☹♞έţίςίғùşίςόνίέħάĻέşħä《》┗╗┏╣┃┃❥◆✌♡♤☄✨⏭⏮￿﹍▁▂▃▄➖⇩“⚔⚠ﮤ❁ツ】∙ღąƴʆɳ↙⚙٭『⇭┗♦❖✱◻◼╚┛┣═╦┳╗➧○〇∆↞【↠❂ɱσɧąɱąđглоток⌚✬ŞƏ⇜┏❣☠━┓↻✾❅➽♚☑️✳️▪️☀️❌➰♨⚜ ⃣➕✏️⚡️⬇️✅⭕️°√☜✘✖✞☛☚؁ٖ-ٖ♗↯⇄⛔️❕⚰⇦⌛✋ℹ@!#$%^&*()_+=|✔ஜ۩۞۩ஜ♥•﴾●♥�❤❤˙…×◙⁰█║▌◄?㋡韩国队规划局·⅛⅞∞٬Ŕ▼ค”◘ž▼↓ĕ♥●•]", " ", text, flags=FLAGS)

class MyTweets(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            with codecs.open(os.path.join(self.dirname, fname),"r",encoding="utf-8") as fp:
                data = json.load(fp)
                for tweet in data:
                    tweetText = tokenize(cleantext(tweet['text']))
                    yield tweetText.split()



