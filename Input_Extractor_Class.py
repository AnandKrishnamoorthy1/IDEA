# -*- coding: utf-8 -*-
"""
Created: Wed Jan 22
Updated: Feb 15
Author: Anand Krishnamoorthy,Gowthamy Renny, Murale Krishna
"""
from Text_Parser_Class import Text_Parser
from NLP_read_constants import Read_Constants
import PyPDF2
import pandas as pd
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
from pdfminer import utils
import pdfminer
import re,os,time,ntpath
from collections import defaultdict
import math

class PDF_Extractor:
    """
    Holds the following functions pdfmetadata_extract-Extracts metadata for pdf, pdf_txt_converter-converts pdf to text format,
    TOC_Extract-Extracts Table of contents, get_Corpus_details-Returns corpus details
    """    

    a=Text_Parser()
    def PDF_Metadata_Extract(self,pdf_input):    
       """
        Input: The url path of the file
        Process: Reads the contents of the file and writes the data into the specified folder
        Output: Writes the contents of URL into the specified folder
       """     
       laparams = LAParams()
       rsrcmgr = PDFResourceManager()
       pdfFileObj = open(pdf_input, 'rb')
       pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
       pdf_info = pdfReader.getDocumentInfo()
       page_num = pdfReader.numPages
       device = PDFPageAggregator(rsrcmgr, laparams=laparams)
       return pdf_info
    
    def get_PDF_Metadata(self,run_id,pdf_input,extract_out,file_type):  
        """
        Input: Run ID, DataObject path, Fileobject path, output write type
        Process: Converts the content of the pdf into a dataframe with page and sub-headers, writes the same into a file
        Output: Dataframe and writes the same into a file
        """        
        ts = round(time.time())
        a=Text_Parser()
        read_constants=Read_Constants()
        
        MetaData=self.PDF_Metadata_Extract(pdf_input)
        
        TXT_METADATA_ID=read_constants.Data_Object_ID+'_'+str(ts)        
        TXT_AUTHOR=MetaData.author if MetaData.author else 'Nil'
        TXT_CREATED_BY=MetaData.creator if MetaData.creator else 'Nil'
        TXT_PRODUCED_BY=MetaData.producer if MetaData.producer else 'Nil'
        TXT_TXT_OBJ_ID=read_constants.Text_Object_ID
        TXT_TXT_OBJ_FILEPATH=extract_out
        TXT_ING_TYP=read_constants.INGESTION_TYPE
        metadata_df=pd.DataFrame([[TXT_METADATA_ID,TXT_AUTHOR,TXT_CREATED_BY,TXT_PRODUCED_BY,TXT_TXT_OBJ_ID,TXT_TXT_OBJ_FILEPATH,TXT_ING_TYP]],columns=["METADATA_ID","AUTHOR","CREATED_BY","PRODUCED_BY","TXT_OBJ_ID","TXT_OBJ_FILEPATH","ING_TYP"])        
        a.writer(metadata_df,extract_out+"_MetaData",file_type)
        print("PDF Metadata extraction completed sucessfully!!!")
        return metadata_df
       
    
    def get_header(self,a):
        """Returnsthe header for the given dataframe index
        """    
        head_key=-999
        for i in range(len(dict_key)):
            if dict_key[i]>a:
                head_key=dict_key[i-1]
                break
        if head_key==-999:
            try:
                return head_dict[dict_key[-1]]
            except:
                return '-999'
        else:
            return head_dict[head_key]
    
    def Process_Extractor(self,extract_df,extract_out,file_type):
        """Function formats the input csv into the format required by further processes.
        Combines text based on page and headers.
        """     
        a=Text_Parser() 
        read_constants=Read_Constants()        

        #extract_df=pd.read_csv(extractor_input,encoding='iso-8859-1')
        extract_df=extract_df[(extract_df["OBJ_TYP"].str.contains("LTText")) & (~extract_df["HEADER"].str.contains("table text"))]
        extract_df=extract_df.reset_index()
        #print(extract_df)
        
        extract_df=extract_df[["PG_NUM","TXT","HEADER","PARA_FLAG","ROW_START"]]
        
        header_lst=[]
        for group,extract_df_row in extract_df.groupby(["PG_NUM"]):
            page_lst=(list(extract_df_row.index))
            header_lst.extend([max(page_lst)])
            
        for i,page_row in extract_df.iterrows():
            if (page_row.HEADER=="Header" or page_row.HEADER=="SubHeader") and (i not in header_lst):
                header_lst.extend([page_row.name])
    
        only_head_df=extract_df[extract_df["HEADER"]=="Header"]
        
        global dict_key
        dict_key=list(only_head_df.index)
        dict_value=list(only_head_df.TXT) 
        
        only_subhead_df=extract_df[extract_df["HEADER"]=="SubHeader"]
        subhead_dict_key=list(only_subhead_df.index)
        subhead_dict_value=list(only_subhead_df.TXT)     
        
        head_subhead_df=extract_df[(extract_df["HEADER"]=="SubHeader") | (extract_df["HEADER"]=="Header")]        
        head_subhead_dict_key=list(head_subhead_df.index)
        
            
        global head_dict
        head_dict={}
        for i in range(len(dict_key)):
            head_dict[dict_key[i]]=dict_value[i]
            
        global subhead_dict
        subhead_dict={}
        for i in range(len(subhead_dict_key)):
            subhead_dict[subhead_dict_key[i]]=subhead_dict_value[i]    
    
        header_lst.sort()
        #extract_df.to_csv(extract_out)
    
        df_rows=[]
        reg_ex_blanks=('[_]+')
        Pat_txt_process=re.compile(reg_ex_blanks)        
        for lst_len in range(len(header_lst)-1):
            
            if header_lst[lst_len]==0 and header_lst[lst_len+1] in head_subhead_dict_key:
                temp_df=(extract_df[header_lst[lst_len]:header_lst[lst_len+1]])            
                pg_num=extract_df.PG_NUM.iloc[header_lst[lst_len]]
                header_txt=self.get_header(header_lst[lst_len])
                header_txt=Pat_txt_process.sub('', header_txt)
                header_txt=header_txt.replace(r'||',r'\n')
                subhead_txt=subhead_dict[header_lst[lst_len]] if header_lst[lst_len] in subhead_dict_key else ''
                subhead_txt=subhead_txt.replace(r'||',r'\n')
                subhead_txt=Pat_txt_process.sub('', subhead_txt)
            
            elif header_lst[lst_len]==0 and header_lst[lst_len+1] not in head_subhead_dict_key:
                temp_df=(extract_df[header_lst[lst_len]:header_lst[lst_len+1]+1])            
                pg_num=extract_df.PG_NUM.iloc[header_lst[lst_len]]
                header_txt=self.get_header(header_lst[lst_len])
                header_txt=Pat_txt_process.sub('', header_txt)
                header_txt=header_txt.replace(r'||',r'\n')
                subhead_txt=subhead_dict[header_lst[lst_len]] if header_lst[lst_len] in subhead_dict_key else ''
                subhead_txt=subhead_txt.replace(r'||',r'\n')  
                subhead_txt=Pat_txt_process.sub('', subhead_txt)
            
            elif header_lst[lst_len] in head_subhead_dict_key and header_lst[lst_len+1] in head_subhead_dict_key:
                temp_df=(extract_df[header_lst[lst_len]+1:header_lst[lst_len+1]])            
                pg_num=extract_df.PG_NUM.iloc[header_lst[lst_len]+1]
                header_txt=self.get_header(header_lst[lst_len])
                header_txt=Pat_txt_process.sub('', header_txt)
                header_txt=header_txt.replace(r'||',r'\n')
                subhead_txt=subhead_dict[header_lst[lst_len]] if header_lst[lst_len] in subhead_dict_key else ''
                subhead_txt=subhead_txt.replace(r'||',r'\n') 
                subhead_txt=Pat_txt_process.sub('', subhead_txt)
                
            elif header_lst[lst_len] in head_subhead_dict_key and header_lst[lst_len+1] not in head_subhead_dict_key:
                temp_df=(extract_df[header_lst[lst_len]+1:header_lst[lst_len+1]+1])              
                pg_num=extract_df.PG_NUM.iloc[header_lst[lst_len]+1]
                header_txt=self.get_header(header_lst[lst_len])
                header_txt=Pat_txt_process.sub('', header_txt)
                header_txt=header_txt.replace(r'||',r'\n')
                subhead_txt=subhead_dict[header_lst[lst_len]] if header_lst[lst_len] in subhead_dict_key else ''
                subhead_txt=str(subhead_txt).replace(r'||',r'\n')   
                subhead_txt=Pat_txt_process.sub('', subhead_txt)
                
            elif header_lst[lst_len] not in head_subhead_dict_key and header_lst[lst_len+1] in head_subhead_dict_key:
                temp_df=(extract_df[header_lst[lst_len]+1:header_lst[lst_len+1]])            
                pg_num=extract_df.PG_NUM.iloc[header_lst[lst_len]+1] 
                header_txt=self.get_header(header_lst[lst_len])
                header_txt=Pat_txt_process.sub('', header_txt)
                header_txt=header_txt.replace(r'||',r'\n')
                subhead_txt=subhead_dict[header_lst[lst_len]] if header_lst[lst_len] in subhead_dict_key else ''
                subhead_txt=subhead_txt.replace(r'||',r'\n')   
                subhead_txt=Pat_txt_process.sub('', subhead_txt)
                
            else:
                temp_df=(extract_df[header_lst[lst_len]+1:header_lst[lst_len+1]+1])            
                pg_num=extract_df.PG_NUM.iloc[header_lst[lst_len]+1]
                header_txt=self.get_header(header_lst[lst_len])
                header_txt=Pat_txt_process.sub('', header_txt)
                header_txt=header_txt.replace(r'||',r'\n')
                subhead_txt=subhead_dict[header_lst[lst_len]] if header_lst[lst_len] in subhead_dict_key else ''
                subhead_txt=subhead_txt.replace(r'||',r'\n')  
                subhead_txt=Pat_txt_process.sub('', subhead_txt)
            
            
            if temp_df.empty:
                continue
                """
                ROW_START=0
                df_rows.append([pg_num,'',header_txt,subhead_txt,ROW_START])
                """
            else:
                prev_i=0
                for i,row in temp_df.iterrows():
                    if i==temp_df.index.min():
                        df_rows.append([pg_num,row.TXT,header_txt,subhead_txt,row.ROW_START])                        
                    elif temp_df.PARA_FLAG.loc[prev_i]:
                        df_rows.append([pg_num,row.TXT,header_txt,subhead_txt,row.ROW_START])
                    elif temp_df.PARA_FLAG.loc[prev_i] and temp_df.PARA_FLAG.loc[i]:
                        df_rows.append([pg_num,row.TXT,header_txt,subhead_txt,row.ROW_START])                        
                    else:
                        df_rows[-1][1]=df_rows[-1][1].rstrip('\n')+row.TXT
                    prev_i=i
                    
            lst_len+=1
        
        head_df=pd.DataFrame(df_rows,columns=["PG_NUM","TXT","HEADER","SUBHEADER","ROW_START"])
        #head_df=head_df[head_df.TXT.str.match(r' ?[0-9]+ ?')== False]
        head_df=head_df[head_df.TXT.str.strip('\n').str.strip().str.isnumeric()== False]
        head_df['TXT_OBJ_ID']=read_constants.Text_Object_ID

        main_sent_start=head_df.ROW_START.min()
        mask=head_df.ROW_START>main_sent_start+3
        head_df['SENT_TYPE_IND']=''
        head_df.loc[mask,'SENT_TYPE_IND']='sub-sentence'
        head_df.loc[~mask,'SENT_TYPE_IND']='Main-sentence'
        
        """New Logic added to remove the bullet points:Starts"""

        Char_bulletin='(^\([a-zA-Z]\) )'
        Num_bulletin='(^\([0-9]\) )'
        Romen_bulletin='(\((ix|iv|v?i{0,3}\)) )'
        Hyphen_bulletin='(|—)'
        quest_bulletin='(\?|)'
        tailing_chars='(–|:|)'
        
        
        for i,row in head_df.iterrows():
            if len(row.TXT)>5:
                str1=str(row.TXT[0:6])
                str2=str(row.TXT[6:])
                str1=re.sub(Char_bulletin, '', str1)
                str1=re.sub(Num_bulletin, '', str1)
                str1=re.sub(Romen_bulletin, '', str1)
                str1=re.sub(Hyphen_bulletin, '', str1)
                str1=re.sub(quest_bulletin, '', str1)
                str_full=str1+str2
                str1=str_full[:-5]
                str2=str_full[-5:]
                str2=re.sub(tailing_chars, '', str2)
                head_df.loc[i,'TXT']=str1+str2        
        """New Logic added to remove the bullet points:Ends"""
        
        
        a.writer(head_df,extract_out,file_type)
        print("Process extractor completed Sucessfully!!")
        return head_df
        
    
    def get_TOC_Extract(self,DOC_FILTER_DF,extract_out,file_type):
        """
        Input: Input dataframe
        Process: Extracts the total pages, list of the pagemumbers,total headers, list of the headers
        Output: Dataframe and writes the same into a file
        """               
        a=Text_Parser()
        pages=DOC_FILTER_DF.PG_NUM.unique()
        page_count=DOC_FILTER_DF.PG_NUM.nunique()
        headers=DOC_FILTER_DF.HEADER.unique()
        headers_count=DOC_FILTER_DF.HEADER.nunique()
        txt_ob_id=DOC_FILTER_DF.TXT_OBJ_ID.iloc[0]
        list_toc=[page_count,pages.tolist(),headers_count,headers.tolist(),txt_ob_id]
        toc_df=pd.DataFrame([list_toc],columns=['TOT_PG_NUM','PG_NUM','TOT_HEADERS_NUM','HEADERS','TXT_OBJ_ID'])        
        #a.writer(toc_df,extract_out,file_type)
        print("Table Of Contents Extraction Completed!!")
        return toc_df

    def get_Corpus_details(self,TOC_df,extract_out,file_type):
        """
        Input: Input Table of content dataframe
        Process: Provides information regarding where each of the corpus(as specified by the user) is written
        Output: Writes the corpus details into a file
        """         
        a=Text_Parser()
        read_constants=Read_Constants()
        
        extractor_filepath, tail = os.path.split(extract_out)
        extractor_filename=ntpath.basename(extract_out)
        text_obj_id=TOC_df.TXT_OBJ_ID.iloc[0]
        corpus_list=[]
        
        corpus_list.append([text_obj_id,extractor_filepath,extractor_filename+'Sentence'])
        corpus_list.append([text_obj_id,extractor_filepath,extractor_filename+'Tokens'])
        corpus_list.append([text_obj_id,extractor_filepath,extractor_filename+'Stem'])
        corpus_list.append([text_obj_id,extractor_filepath,extractor_filename+'Lemma'])
        corpus_list.append([text_obj_id,extractor_filepath,extractor_filename+'Tokens_Cleansed'])
        
        if read_constants.SENT_CATEGORIZE==1:corpus_list.append([text_obj_id,extractor_filepath,extractor_filename+'Sentence_Category'])
        if read_constants.SENT_TENSE==1:corpus_list.append([text_obj_id,extractor_filepath,extractor_filename+'Sentence_Tense'])
        if read_constants.POS_FLAG==1:corpus_list.append([text_obj_id,extractor_filepath,extractor_filename+'POSTagging'])
        if read_constants.NE_FLAG==1:corpus_list.append([text_obj_id,extractor_filepath,extractor_filename+'Named_Entity'])
        if read_constants.REL_FLAG==1:corpus_list.append([text_obj_id,extractor_filepath,extractor_filename+'Entity_Relationship'])
        if read_constants.NGRAM_FLAG==1:corpus_list.append([text_obj_id,extractor_filepath,extractor_filename+'Ngrams'])
        
        corpus_df=pd.DataFrame(corpus_list,columns=['TXT_OBJ_ID','CORPUS_FILEPATH','CORPUS_NM'])
        #a.writer(corpus_df,extract_out+'Corpus_details',file_type)
        print("Corpus details Extraction Completed!!")
        return corpus_df

    def MAS_FOOTER_RM(self,new_format_coord_df):
        """!!!!New Logic Added for footer removal in MAS !!!!"""
        #new_format_coord_df.to_csv(r"C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\Input\New_MAS\temp\test_mas639_With_Footer.csv",index=False)
        new_format_coord_df_cord=pd.DataFrame(columns=["TXT","COLUMN_BOTTOM","Time","PG_NUM","OBJ_TYP","LINE_No","OBJ_SEQ","SUB_OBJ_SEQ","ROW_START","ROW_END","COLUMN_TOP","PAGE_OBJ_SEQ","LINE_OBJ_SEQ","END_FLAG","HEIGHT","WIDTH","HTYPE","HEADER"])
        for grp, df_grp in new_format_coord_df.groupby(['PG_NUM']):
        
            try:
                line_no=9999
                #line_no=df_grp[(df_grp.WIDTH>140) & (df_grp.WIDTH<150) & (df_grp.ROW_START>70) & (df_grp.ROW_START<75) & (df_grp.ROW_END>210) & (df_grp.ROW_END<225) & (df_grp.COLUMN_BOTTOM<200) & (df_grp.HEIGHT<2)].LINE_No
                line_no=df_grp[(df_grp.WIDTH>140) & (df_grp.WIDTH<150) & (df_grp.COLUMN_BOTTOM<400) & (df_grp.HEIGHT<2)].LINE_No
                line_no=int(line_no) 
                #print("!!!!!!!!!!!!Line No!!!!!!!!!!!!!",line_no)
                df_grp=df_grp[df_grp.LINE_No<line_no]
                new_format_coord_df_cord=pd.concat([new_format_coord_df_cord, df_grp], ignore_index=True)
            except:
                new_format_coord_df_cord=pd.concat([new_format_coord_df_cord, df_grp], ignore_index=True)
        
        #new_format_coord_df_cord.to_csv(r"C:\Users\bfsbicoe14\Desktop\TEXT_ANALYSIS\NLP Text Analytics\Input\Input\Input\New_MAS\temp\test_mas639.csv",index=False)
        return new_format_coord_df_cord
    
    def pdf_metadata_report_parser(self,input_fp,seq_out_fp):
        seq_out_fp=seq_out_fp+"_with_sequence.csv"		
        
        #Read the PDF extract the page Layout
        def extract_layout_by_page(pdf_path):
            """
            Extracts LTPage objects from a pdf file.
            
            slightly modified from
            https://euske.github.io/pdfminer/programming.html
            """
            laparams = LAParams()
        
            fp = open(pdf_path, 'rb')
            parser = PDFParser(fp)
            document = PDFDocument(parser)
        
            if not document.is_extractable:
                raise PDFTextExtractionNotAllowed
        
            rsrcmgr = PDFResourceManager()
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
        
            layouts = []
            headings_para_index=[]
            for page in PDFPage.create_pages(document):
                interpreter.process_page(page)
                layouts.append(device.get_result())
                layout = device.get_result()
                page_id=layout.pageid
                pattern = re.compile("<([a-zA-Z]+)")
          
              
                headings=[]
                
                txt_line_cnt=0
                key_value=[]
                key=""
                value=""
            
            #Extract all pdf objects sorted with coordinates and in sequence
                for obj in layout:    
                    pattern_match = (pattern.findall(str(obj)))
                    if isinstance(obj, pdfminer.layout.LTTextBox): 
                        txt_line_cnt=0
                 
                        for item in obj :
                            pattern_match1= (pattern.findall(str(item)))              
                            heading=False
                            if isinstance(item, pdfminer.layout.LTTextLine) :
                                for characters in item :
                                    if isinstance(characters, pdfminer.layout.LTChar) :
                                        #print("font",characters.fontname,"size",str(characters.size),"color",characters.color)
                                        if 'Bold' in str(characters.fontname) :
                                            heading=True
                                        else:
                                            heading=False
                            if heading==True:
                                hbbox=[str(page_id),pattern_match1,"Head", str(obj.index),str(txt_line_cnt),item.bbox[0], item.bbox[1],item.bbox[2],item.bbox[3],item.bbox[2]-item.bbox[0],item.bbox[3]-item.bbox[1], str(item.get_text()),obj.bbox[0],obj.bbox[2],'','']
                                headings.append(hbbox)
                                headings_para_index.append(hbbox)
                                
                            else :
                                hbbox=[str(page_id),pattern_match1,"Para", str(obj.index),str(txt_line_cnt),item.bbox[0], item.bbox[1],item.bbox[2],item.bbox[3],item.bbox[2]-item.bbox[0],item.bbox[3]-item.bbox[1], str(item.get_text()),obj.bbox[0],obj.bbox[2],'','']  
                                headings_para_index.append(hbbox)
                                
                            txt_line_cnt+=1
                    else:
                        if isinstance(obj, pdfminer.layout.LTRect) or isinstance(obj, pdfminer.layout.LTCurve) or  isinstance(obj, pdfminer.layout.LTFigure):
                            obj_index=""
                        else:
                            obj_index=str(obj.index)
                        fulltext=""
                        fontname=''
                        fonttext=''
                        Bold_Text=''
                        hbbox=[str(page_id),pattern_match,"Others", obj_index,str(txt_line_cnt),obj.bbox[0], obj.bbox[1],obj.bbox[2],obj.bbox[3],obj.bbox[2]-obj.bbox[0],obj.bbox[3]-obj.bbox[1],fulltext,obj.bbox[0],obj.bbox[2],'','']
                        headings_para_index.append(hbbox)
            
                """
                Get the X and Y range as  sorted by the value and converted to int, right now x0 and y0 are used, if needed can be tweaked for adj_x0 and adj_y0
                """
                x_range=sorted(set([int(x0) for pageid,item,htype,tbox,tline,x0,y0,x1,y1,w,h,txt,tb_x0,tb_x1,font,csize  in headings_para_index]))
                y_range=sorted(set([int(y0) for pageid,item,htype,tbox,tline,x0,y0,x1,y1,w,h,txt,tb_x0,tb_x1,font,csize  in headings_para_index]),reverse=True)
            
                """
                Sequence the row and column data
                """
              
                head_para=[]
                for y_order,y in enumerate(y_range) :        
                    for x_order,x in enumerate(x_range) :  
                        for pageid,item,htype,tbox,tline,x0,y0,x1,y1,w,h,txt,tb_x0,tb_x1,font,csize  in headings_para_index :
                             
                            if int(y0)==y and int(x0)==x:
                                orig_x=int(x0)
                                orig_y=int(y0)
                                head_para.append((htype,str(y_order),str(x_order),txt,tbox,orig_y,orig_x))
        
            return layouts,headings_para_index
        
        page_layouts,headings_para_index = extract_layout_by_page(input_fp) 
        
        def get_min_max_table(pagenum):
         
                    #y0_range=sorted(([(y0) for pageid,item,line_no,tbox,tline,x0,x1,y0,y1,txt,page_seq,line_seq,end_flag,h,w,header in new_set if pageid==str(pagenum) and 'rect' in item.lower()]),reverse=True)
                    r_y0_range=sorted(set([(y0) for pageid,item,htype,tbox,tline,x0,y0,x1,y1,w,h,txt,tb_x0,tb_x1,font,csize in headings_para_index if pageid==str(pagenum) and 'rect' in str(item).lower()]),reverse=True)            
                    r_y1_range=sorted(set([(y1) for pageid,item,htype,tbox,tline,x0,y0,x1,y1,w,h,txt,tb_x0,tb_x1,font,csize in headings_para_index if pageid==str(pagenum) and 'rect' in str(item).lower()]),reverse=True)            
                    r_x0_range=sorted(set([(x0) for pageid,item,htype,tbox,tline,x0,y0,x1,y1,w,h,txt,tb_x0,tb_x1,font,csize in headings_para_index if pageid==str(pagenum) and 'rect' in str(item).lower()]))
                    r_x1_range=sorted(set([(x1) for pageid,item,htype,tbox,tline,x0,y0,x1,y1,w,h,txt,tb_x0,tb_x1,font,csize in headings_para_index if pageid==str(pagenum) and 'rect' in str(item).lower()]))
        
                    if (len(r_y0_range)==0):
                        rmin_y0=0
                    else:
                        rmin_y0=min(r_y0_range)
                        
                        
                    if (len(r_x0_range)==0):
                        rmin_x0=0
                    else:
                        rmin_x0=min(r_x0_range)
        
                    if (len(r_y1_range)==0):
                        rmax_y1=0
                    else:
                        rmax_y1=max(r_y1_range)
        
                    if (len(r_x1_range)==0):
                        rmax_x1=0
                    else:
                        rmax_x1=max(r_x1_range)
                    
                    if rmin_x0!=0 and rmin_y0!=0 and rmax_x1!=0 and rmax_y1!=0:
                        rect_present_flag='True'
                        """MAS specific code to remove the exceptions noticed in forming tables Starts."""
                        dif=rmax_x1-rmin_x0
                        if dif<430:
                            rect_present_flag='False'
                        """MAS specific code to remove the exceptions noticed in forming tables Ends."""
                    else:
                        rect_present_flag='False'
                    #print("x,y,x1,y1",rmin_x0,rmin_y0,rmax_x1,rmax_y1)
                    
                    print("Page No: ",pagenum)
                    print("X Min: ",rmin_x0,"X Max: ",rmax_x1)
                    print("y Min: ",rmin_y0,"y Max: ",rmax_y1)
                    print("Contains rect:",rect_present_flag)
                    
                    return rmin_x0,rmin_y0,rmax_x1,rmax_y1,rect_present_flag
        
                   
        def text_within_rect(x0,y0,x1,y1,rect_min_x0,rect_min_y0,rect_max_x1,rect_max_y1):
             if int(x0)>= int(rect_min_x0) and int(x1) <= int(rect_max_x1) and int(y0)>= int(rect_min_y0) and int(y1)<=int(rect_max_y1):
                 text_in_table='text in table'
             else:
                 text_in_table='normal text'
             return text_in_table             
             
        x0_range=[]
        x1_range=[]
        new_coord=[]
        box_sequence=0
        new_set_list = []
        
        
        page_order=sorted(set([pageid for pageid,item,htype,tbox,tline,x0,y0,x1,y1,w,h,txt,tb_x0,tb_x1,font,csize in headings_para_index]))
        
        """
           For each page, sort the objects based on y0 and find the end object for that line (end coordinates,object sequence in a page,object sequence in a line etc)
        """
        
        for pagenum in page_order:
        
            rect_min_x0,rect_min_y0,rect_max_x1,rect_max_y1,rect_present_flag=get_min_max_table(pagenum)    
            
            page_box_sequence=0
            line_number=0
            #print(pagenum)
            y_range=sorted(set([int(y0) for pageid,item,htype,tbox,tline,x0,y0,x1,y1,w,h,txt,tb_x0,tb_x1,font,csize in headings_para_index if pageid==str(pagenum)]),reverse=True)
            #print(y_range)
            ##For each Y0
            for y in y_range:
                x0_range=sorted(set([(x0) for pageid,item,htype,tbox,tline,x0,y0,x1,y1,w,h,txt,tb_x0,tb_x1,font,csize in headings_para_index if int(y0)==y and pageid==str(pagenum)]))
                x1_range=sorted(set([(x1) for pageid,item,htype,tbox,tline,x0,y0,x1,y1,w,h,txt,tb_x0,tb_x1,font,csize in headings_para_index if int(y0)==y and pageid==str(pagenum)]))
                least_x0=min(x0_range)
                max_x1=max(x1_range)
                line_box_sequence=0  
                line_number+=1
                ##For each(x1) object corresponding to Y0
                
                for pageid,item,htype,tbox,tline,x0,y0,x1,y1,w,h,txt,tb_x0,tb_x1,font,csize in headings_para_index:
                    head_type='None'
                    table_text='None'
                    if int(y0)==y and pageid==pagenum:
                        
                        if int(x1) < int(max_x1):
                            end_flag='False'
                            page_box_sequence+=1
                            line_box_sequence+=1
                            height=abs(y0-y1)
                            width=abs(x0-x1)
                            
                            if "text" in str(item).lower() and rect_present_flag=='True':
                                table_text=text_within_rect(x0,y0,x1,y1,rect_min_x0,rect_min_y0,rect_max_x1,rect_max_y1)
                                                      
                            if table_text=="text in table":    
                                head_type='table text'#new_set=[pageid,item,line_number,tbox,tline,x0,x1,y0,y1,txt,page_box_sequence,line_box_sequence,end_flag]
                            else:              
                                if htype=='Head' and txt.strip() != "||" and 'text' in str(item).lower():
                                    head_type='Highlights'  
                                else:
                                    head_type='None'
                            
                            new_set=str(pageid) + "," + str(item) + "," + str(line_number) + "," + str(tbox) + "," + str(tline) + "," + str(x0) + "," + str(x1) + "," + str(y0) + "," + str(y1) + "," + str(txt) + "," + str(page_box_sequence) + "," + str(line_box_sequence) + "," + str(end_flag) + "," + str(height) + "," + str(width) +"," + htype +"," +head_type
                            new_set_list1 = [str(pageid), str(item),str(line_number), str(tbox) , str(tline), x0 ,x1, y0,y1, str(txt) , str(page_box_sequence) , str(line_box_sequence), str(end_flag), str(height), str(width), htype ,head_type]
                            new_set_list.append(new_set_list1)
                        elif int(x1)==int(max_x1):
                            end_flag='True'
                            page_box_sequence+=1
                            line_box_sequence+=1
                            height=abs(y0-y1)
                            width=abs(x0-x1)
                            
                            if 'text' in str(item).lower() and rect_present_flag=='True':
                                table_text=text_within_rect(x0,y0,x1,y1,rect_min_x0,rect_min_y0,rect_max_x1,rect_max_y1)
                            ###To get proper headers and subheaders ####
                            if table_text=="text in table":    
                                head_type='table text'
                            else:
                                if htype=='Head'  and 'text' in str(item).lower():
                                    if str(txt.strip()).isupper() and txt.strip() != "||":
                                        head_type='Header' 
                                    elif txt.strip() != "||" and not str(txt.strip()).isupper() and not str(txt.strip()).islower() :
                                        head_type='SubHeader'
                                    else:
                                        head_type='None'
                            
                            #new_set=str(pageid) + "," + str(item) + "," + str(line_number) + "," + str(tbox) + "," + str(tline) + "," + str(x0) + "," + str(x1) + "," + str(y0) + "," + str(y1) + "," + str(txt) + "," + str(page_box_sequence) + "," + str(line_box_sequence) + "," + str(end_flag) + "," + str(height) + "," + str(width)+ "," + htype+"," +head_type
                            new_set_list1 = [str(pageid), str(item),str(line_number), str(tbox) , str(tline), x0 ,x1, y0,y1, str(txt) , str(page_box_sequence) , str(line_box_sequence), str(end_flag), str(height), str(width), htype ,head_type]
                            new_set_list.append(new_set_list1)
                        else: 
                            #None
                            new_set_list1 = [str(pageid), str(item),str(line_number), str(tbox) , str(tline), x0 ,x1, y0,y1, str(txt) , str(page_box_sequence) , str(line_box_sequence), str(end_flag), str(height), str(width), htype ,head_type]
                            new_set_list.append(new_set_list1)
							
                        new_coord_df=pd.DataFrame(new_set_list,columns=["PG_NUM","OBJ_TYP","LINE_No","OBJ_SEQ", "SUB_OBJ_SEQ", "ROW_START","ROW_END","COLUMN_BOTTOM","COLUMN_TOP","TXT","PAGE_OBJ_SEQ","LINE_OBJ_SEQ","END_FLAG","HEIGHT","WIDTH","HTYPE", "HEADER"])
                        
                        
                        """New code added to remove Header and Footer"""
                        """
                        Sentences1_freq=new_coord_df['COLUMN_BOTTOM'].groupby(new_coord_df['TXT']).value_counts().sort_values(ascending=False).reset_index(name="Time")
                        Sentences1_freq_all = pd.merge(Sentences1_freq, new_coord_df, how='left', left_on=['COLUMN_BOTTOM','TXT'], right_on = ['COLUMN_BOTTOM','TXT'])
                        Sentences1_freq_all_1 = Sentences1_freq_all.loc[Sentences1_freq_all['Time'] == 1]
                        Sentences1_freq_all_2=Sentences1_freq_all_1.loc[Sentences1_freq_all.TXT.str.match(r'INTERNATIONAL MONETARY FUND  [\d]+ \|\|') == False]
                        Sentences1_freq_all_3=Sentences1_freq_all_2.loc[Sentences1_freq_all.TXT.str.match(r'[\d]+ \|\|')== False]
                        Sentences1_freq_all_4=Sentences1_freq_all_3.loc[Sentences1_freq_all.TXT.str.match( r'\|\|' ) == False]
                        Sentences1_freq_all_4.LINE_No = Sentences1_freq_all_4.LINE_No.astype('int64')
                        Sentences1_freq_all_4.PG_NUM = Sentences1_freq_all_4.PG_NUM.astype('int64')
                        
                        new_coord_df=Sentences1_freq_all_4.sort_values(['PG_NUM', 'LINE_No','COLUMN_BOTTOM'])
                        """
                        """New code ends! """


                        new_coord_df.LINE_No = new_coord_df.LINE_No.astype('int64')
                        new_coord_df.PG_NUM = new_coord_df.PG_NUM.astype('int64')                       
                        new_coord_df.COLUMN_BOTTOM = new_coord_df.COLUMN_BOTTOM.astype('float')
                        new_coord_df['Time']=1
                        new_coord_df=new_coord_df.sort_values(['PG_NUM', 'LINE_No','COLUMN_BOTTOM'])
        #Recursively extract the character from the text objects, to match up with table columns
        
        TEXT_ELEMENTS = [
            pdfminer.layout.LTTextBox,
            pdfminer.layout.LTTextBoxHorizontal,
            pdfminer.layout.LTTextLine,
            pdfminer.layout.LTTextLineHorizontal
        ]
        
        def flatten(lst):
            """Flattens a list of lists"""
            return [subelem for elem in lst for subelem in elem]
        
        
        def extract_characters(element):
            """
            Recursively extracts individual characters from 
            text elements. 
            """
            if isinstance(element, pdfminer.layout.LTChar):
                return [element]
        
            if any(isinstance(element, i) for i in TEXT_ELEMENTS):
                return flatten([extract_characters(e) for e in element])
        
            if isinstance(element, list):
                return flatten([extract_characters(l) for l in element])
        
            return []
        
        def width(rect):
            x0, y0, x1, y1 = rect.bbox
            return min(x1 - x0, y1 - y0)
        
        def area(rect):
            x0, y0, x1, y1 = rect.bbox
            return (x1 - x0) * (y1 - y0)
        
        
        def cast_as_line(rect):
            """
            Replaces a retangle with a line based on its longest dimension.
            """
            x0, y0, x1, y1 = rect.bbox
        
            if x1 - x0 > y1 - y0:
                return (x0, y0, x1, y0, "H")
            else:
                return (x0, y0, x0, y1, "V")
        
        def num_rows(group):
            return len(group)
        
        def num_columns(group):
            return len(group[0])
        
        
        def does_it_intersect(x, xmin, xmax):
            return (x <= xmax and x >= xmin)
        
        def find_bounding_rectangle(x, y, lines):
            """
            Given a collection of lines, and a point, try to find the rectangle 
            made from the lines that bounds the point. If the point is not 
            bounded, return None.
            """
            
            v_intersects = [l for l in lines
                            if l[4] == "V"
                            and does_it_intersect(y, l[1], l[3])]
        
            h_intersects = [l for l in lines
                            if l[4] == "H"
                            and does_it_intersect(x, l[0], l[2])]
        
            if len(v_intersects) < 2 or len(h_intersects) < 2:
                return None
        
            v_left = [v[0] for v in v_intersects
                      if v[0] < x]
        
            v_right = [v[0] for v in v_intersects
                       if v[0] > x]
        
            if len(v_left) == 0 or len(v_right) == 0:
                return None
        
            x0, x1 = max(v_left), min(v_right)
        
            h_down = [h[1] for h in h_intersects
                      if h[1] < y]
        
            h_up = [h[1] for h in h_intersects
                    if h[1] > y]
        
            if len(h_down) == 0 or len(h_up) == 0:
                return None
        
            y0, y1 = max(h_down), min(h_up)
        
            return (x0, y0, x1, y1)
        
        
        # Mention the page number you want to extract the table from
        for page in page_layouts:
            current_page = page
        
            #Separate texts and rect elements
            texts = []
            rects = []
            table_content = []
            
            # seperate text and rectangle elements
            for e in current_page:
                if isinstance(e, pdfminer.layout.LTTextBoxHorizontal):
                    texts.append(e)
                elif isinstance(e, pdfminer.layout.LTRect):
                    rects.append(e)
        
            characters = extract_characters(texts)
            
            #Replace rectangle with line
            lines = [cast_as_line(r) for r in rects
                     if width(r) < 2 and
                     area(r) > 1]
            #Choosing the bottom left corner,top right corner and centre to find the cahracter and cell
            box_char_dict = {}
            
            for c in characters:
                # choose the bounding box that occurs the majority of times for each of these:
                bboxes = defaultdict(int)
                l_x, l_y = c.bbox[0], c.bbox[1]
                bbox_l = find_bounding_rectangle(l_x, l_y, lines)
                bboxes[bbox_l] += 1
            
                c_x, c_y = math.floor((c.bbox[0] + c.bbox[2]) / 2), math.floor((c.bbox[1] + c.bbox[3]) / 2)
                bbox_c = find_bounding_rectangle(c_x, c_y, lines)
                bboxes[bbox_c] += 1
            
                u_x, u_y = c.bbox[2], c.bbox[3]
                bbox_u = find_bounding_rectangle(u_x, u_y, lines)
                bboxes[bbox_u] += 1
            
                # if all values are in different boxes, default to character center.
                # otherwise choose the majority.
                if max(bboxes.values()) == 1:
                    bbox = bbox_c
                else:
                    bbox = max(bboxes.items(), key=lambda x: x[1])[0]
            
                if bbox is None:
                    continue
            
                if bbox in box_char_dict.keys():
                    box_char_dict[bbox].append(c)
                    continue
            
                box_char_dict[bbox] = [c]
            
            #To capture empty cells, I choose a grid on points across the page and try to assign them to a cell. If this cell isn't present in box_char_dict, then it is created and left empty.
            xmin, ymin, xmax, ymax = current_page.bbox
            
            for x in range(int(xmin), int(xmax), 10):
                for y in range(int(ymin), int(ymax), 10):
                    bbox = find_bounding_rectangle(x, y, lines)
            
                    if bbox is None:
                        continue
            
                    if bbox in box_char_dict.keys():
                        continue
            
                    box_char_dict[bbox] = []
            
            #All that remains is to map between the ordering of cells on the page and a python data structure and between the ordering of characters in a cell and a string.
            def chars_to_string(chars):
                """
                Converts a collection of characters into a string, by ordering them left to right, 
                then top to bottom.
                """
                if not chars:
                    return ""
                rows = sorted(list(set(c.bbox[1] for c in chars)), reverse=True)
                text = ""
                for row in rows:
                    sorted_row = sorted([c for c in chars if c.bbox[1] == row], key=lambda c: c.bbox[0])
                    text += "".join(c.get_text().replace(",","|")  for c in sorted_row)
                return text
            
            
            def boxes_to_table(box_record_dict):
                """
                Converts a dictionary of cell:characters mapping into a python list
                of lists of strings. Tries to split cells into rows, then for each row 
                breaks it down into columns.
                """
                boxes = box_record_dict.keys()
                rows = sorted(list(set(b[1] for b in boxes)), reverse=True)
                table = []
                for row in rows:
                    sorted_row = sorted([b for b in boxes if b[1] == row], key=lambda b: b[0])
                    table.append([chars_to_string(box_record_dict[b]) for b in sorted_row])
                return table      
            
            tables = boxes_to_table(box_char_dict)
            #print(tables)
            table_content.append(tables)
        
            pattern = re.compile("([0-9])")
            pattern_match = (pattern.findall(str(page)))
            page_no = pattern_match
            prev_col_length=0
            i=0
            j=0
            for i in range(len(tables)):
                for j in range(len(tables[i])):
                    col_length=len(tables[i])
                    if prev_col_length < col_length:
                        prev_col_length=col_length
            #print("m*n",(str(len(tables))+"*"+str(prev_col_length)))
            row = (len(tables))
            column = (prev_col_length)
            if row>0 and column>0 :
                csv_nm = seq_out_fp+page_no[0]+page_no[1]+"table_pdf.csv"
                df1=pd.DataFrame.from_records(tables)
                
                # Create a Pandas Excel writer using XlsxWriter as the engine.
                writer = pd.ExcelWriter(csv_nm+'neat_output.xlsx', engine='xlsxwriter')
                
                # Convert the dataframe to an XlsxWriter Excel object.
                df1.to_excel(writer, sheet_name='Sheet1', index=False)
                
                workbook  = writer.book
                worksheet = writer.sheets['Sheet1']
                
                # Add some cell formats.
                format1 = workbook.add_format({'text_wrap': True})
                
                # Set the column width and format.
                worksheet.set_column('A:ZZ', 18, format1)
                worksheet.set_row(0,0)
        
        """New Logic Added For Text Formatting"""        
        inp_obj_lst=[]
        new_coord_df = new_coord_df.dropna(subset=['TXT'])
        new_coord_df=new_coord_df.sort_values(['PG_NUM', 'LINE_No','COLUMN_BOTTOM','ROW_START'])               
        
        for grp, df in new_coord_df.groupby(['PG_NUM', 'LINE_No','COLUMN_BOTTOM']):
            TXT=''.join(str(TXT) for TXT in df.TXT)
            inp_obj_lst.append([TXT,df.COLUMN_BOTTOM.iloc[0],df.Time.iloc[0],df.PG_NUM.iloc[0],df.OBJ_TYP.iloc[0],df.LINE_No.iloc[0],df.OBJ_SEQ.iloc[0],df.SUB_OBJ_SEQ.iloc[0],df.ROW_START.iloc[0],df.ROW_END.iloc[0],df.COLUMN_TOP.iloc[0],df.PAGE_OBJ_SEQ.iloc[0],df.LINE_OBJ_SEQ.iloc[0],df.END_FLAG.iloc[0],df.HEIGHT.iloc[0],df.WIDTH.iloc[0],df.HTYPE.iloc[0],df.HEADER.iloc[0]])
                    
        new_format_coord_df=pd.DataFrame(inp_obj_lst,columns=["TXT","COLUMN_BOTTOM","Time","PG_NUM","OBJ_TYP","LINE_No","OBJ_SEQ","SUB_OBJ_SEQ","ROW_START","ROW_END","COLUMN_TOP","PAGE_OBJ_SEQ","LINE_OBJ_SEQ","END_FLAG","HEIGHT","WIDTH","HTYPE","HEADER"])
        
        new_format_coord_df.LINE_No = new_format_coord_df.LINE_No.astype('int64')
        new_format_coord_df.WIDTH = new_format_coord_df.WIDTH.astype('float')
        new_format_coord_df.ROW_START = new_format_coord_df.ROW_START.astype('float')
        new_format_coord_df.ROW_END = new_format_coord_df.ROW_END.astype('float')
        new_format_coord_df.COLUMN_BOTTOM = new_format_coord_df.COLUMN_BOTTOM.astype('float')
        new_format_coord_df.HEIGHT = new_format_coord_df.HEIGHT.astype('float')
               
        """!!!!Calling Function for footer removal in MAS !!!!"""
        new_format_coord_df_cord=self.MAS_FOOTER_RM(new_format_coord_df)
        
        new_format_coord_df_cord=new_format_coord_df_cord[new_format_coord_df_cord.TXT.str.strip(' ').str.strip('\n')!='']
        
        new_format_coord_df_cord['PARA_FLAG']='N'
        
        
        new_format_coord_df_cord.PARA_FLAG=(new_format_coord_df_cord.HEADER=='None') & (new_format_coord_df_cord.TXT.str[-7:].str.contains('; |– |\. |\: |— ')) & (new_format_coord_df_cord.TXT.str[-7:].str.contains('\n')) & (~new_format_coord_df_cord.TXT.str[-7:].str.contains('e.g.'))
        """New Logic Added For Text Formatting ends!!"""
        
        new_format_coord_df_cord.to_csv(seq_out_fp,index=False)               
        return new_format_coord_df_cord

    
class Doc_Extractor:    
    """Yet to develop the code"""
        
class Text_Extractor:
    """Yet to develop the code"""
        
class CSV_Extractor:
    """Yet to develop the code"""
        

class Selection_Filter:
    """" Provides Pagewise or regular expression based filter of the original text dataframe """

    def Page_Selection_Filter(self,text_df,filter_type,filter_start,filter_end,extract_out,file_type):
        """
        Input: Input Text dataframe,filter_type,filter type, filter start, filter end, output destination and the output write type
        Process: Page start and the page end is specified, so the original dataframe is sliced and only the dataframe with thespecified range is retained
        Output: Returns the filtered dataframe, writes the same into a file
        """         
        
        a=Text_Parser()
    #Page Extraction module
        if filter_start<=filter_end:
            df_start=text_df[text_df.PG_NUM==filter_start]
            start_idx=df_start.index.values.min()
            df_end=text_df[text_df.PG_NUM==filter_end]
            end_idx=df_end.index.values.max()
            a.writer(text_df[start_idx:end_idx+1],extract_out,file_type)
            print("Exraction by "+filter_type.upper()+" Completed Sucessfully!!!!")
            return (text_df[start_idx:end_idx+1])
         
    def Regex_Selection_filter(self,text_df,filter_type,filter_start,filter_end,extract_out,file_type): 
        """
        Input: Input Text dataframe,filter_type,filter type, filter start, filter end, output destination and the output write type
        Process: The original dataframe is sliced and the program returns the dataframe with specifed start and End pattern
        Output: Returns the filtered dataframe, writes the same into a file
        """         
        
        a=Text_Parser()
        #Regex Extraction module
        for i,row in text_df.iterrows():
            reg_ex_start=re.search(filter_start+'.*',row.TXT)
                        
            if reg_ex_start:
                reg_ex_start_idx=i
                text_df.loc[i,'TXT']=reg_ex_start.group()

            reg_ex_end=re.search('.*'+filter_end,text_df.TXT.iloc[i])

            if reg_ex_end:
                reg_ex_end_idx=i
                text_df.loc[i,'TXT']=reg_ex_end.group()                
                break
            
        a.writer(text_df[reg_ex_start_idx:reg_ex_end_idx+1],extract_out,file_type)        
        print("Exraction by "+filter_type.upper()+" Completed Sucessfully!!!!")
        return (text_df[reg_ex_start_idx:reg_ex_end_idx+1])
            