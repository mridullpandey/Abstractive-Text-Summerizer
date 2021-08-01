#!/usr/bin/env python
# coding: utf-8

# In[51]:


import unidecode
import re
import nltk 
from bs4 import BeautifulSoup
from mapping import contraction_mapping


# In[52]:


class TextCleaner():
    def lower_case(self,text):
        """Converts text to lowercase"""
        text = text.lower()
    
        return text
    
    def remove_whitespace(self,text):
        """Removes whitespace"""
        text = text.strip()
    
        return " ".join(text.split())
    
    def remove_appostrophe(self,text):
        """Removes apostrophe"""
        text = re.sub(r"'s\b'", '', text)

        return text
    
    def remove_parenthesis_text(self, text):
        """Removes text between parenthesis"""
        text = re.sub(r'\([^)]*\)', '', text)

        return text
    
    def remove_html_tags(self, text):
        """Removes HTML tags from text"""
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")

        return text 
    
    def remove_special_chars(self, text):
        """Removes special characters"""
        text = re.sub(r'[^a-zA-Z]', ' ', text)

        return text
    
    def remove_accented_chars(self, text):
        """remove accented characters from text, e.g. café"""
        text = unidecode.unidecode(text)

        return text
    
    def remove_stop_words(self, text):
        """remove stop words"""
        stop_words = set(nltk.corpus.stopwords.words('english'))
        text = ' '.join([w for w in text.split() if not w in stop_words])

        return text
    
    def remove_short_words(self, text):
        """removes short words"""
        text = ' '.join([w for w in text.split() if len(w) > 1])

        return text
    
    def remove_contractions(self, text):
        '''maps contractions to full form'''
        text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])

        return text
    
    def text_preprocessing(self, text, lower = True,contractions = True, whitespace = True, appostrophes = True,
                      parenthesis_text = True, html_tags = True, special_chars = True, 
                      accented_chars = True, stop_words = True, short_words = True, 
                      stories_tags = True, highlights_tags = True, short_form = True):
        """
        Returns clean text by doing 
        the following operations: 

        1.  Lowercase
        2.  Contraction Mapping 
        3.  Remove whitespace
        4.  Remove 's
        5.  Remove anything between brackets
        6.  Remove HTML tags
        7.  Remove special characters
        8.  Remove Accented words
        9.  Remove stopwords
        10. Remove shortwords

        """

        if lower:
            text = self.lower_case(text)

        if whitespace:
            text = self.remove_whitespace(text)

        if appostrophes:
            text = self.remove_appostrophe(text)

        if parenthesis_text:
            text = self.remove_parenthesis_text(text)

        if html_tags:
            text = self.remove_html_tags(text)

        if special_chars:
            text = self.remove_special_chars(text)

        if accented_chars:
            text = self.remove_accented_chars(text)

        if stop_words:
            text = self.remove_stop_words(text)

        if short_words:
            text = self.remove_short_words(text)

        if short_form:
            text = self.remove_contractions(text)

        return text

