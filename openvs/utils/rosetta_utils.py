from __future__ import print_function
import os,sys
import numpy as np
import pandas as pd
import subprocess
import gzip

def _check_tags_ndx(fn,tags, line_marker = "nomarker", tags_marker = "description"):

    indices = []
    valid_tags = []
    for line in file(fn):
        if line_marker == "nomarker": #read every line
            pass
        elif not line.startswith(line_marker):
            continue
        if tags_marker in line:
            fields = line.strip().split()
            for tag in tags:
                try:
                    indices.append(fields.index(tag))
                    valid_tags.append(tag)
                except ValueError:
                    pass
                    #print("Cannot find tag:%s in file, skip it."%tag)
            break
    return valid_tags,indices

def _check_tags_ndx_line(line, tags):

    indices = []
    valid_tags = []

    fields = line.strip().split()
    for tag in tags:
        try:
            if tag == "description":
                indices.append(-1)
                valid_tags.append(tag)
            else:
                indices.append(fields.index(tag))
                valid_tags.append(tag)
        except ValueError:
            pass

    return valid_tags,indices

def _line_parser(line,indices=None,dtypes=(float,) ):
    values = []
    if indices is None:
        return line
    else:
        fields = line.strip().split()
        if 'FAILURE' in fields[-2]:
            return False
        for i in indices:
            if len(dtypes) == 1: 
                dtype = dtypes[0]
            else:
                dtype = dtypes[indices.index(i)]
            try:
                if 'nan' in fields[i]: return False
                values.append(dtype(fields[i]))
            except (IndexError, ValueError):
                print(fields, i, dtype)
                try:
                    fields[i]
                except IndexError:
                    continue
                
                raise Exception("Cannot convert %s to %s"%(fields[i],dtype))
                
            
        if len(values) != len(indices):
            return False
        return values


def read_log_fstream(infh, tags, dtypes, line_marker = "SCORE:" , tags_marker = "description", header_line=None, ignore_incomplete=False, verbose=False):
    valid_tags = []
    values = []
    if header_line is not None:
        valid_tags, indices = _check_tags_ndx_line(header_line, tags)
        assert len(valid_tags) == len(tags), "Valid tags: %s doesn't match specified tags %s!"%(valid_tags,tags)
        if valid_tags[-1] == "description":
            indices[-1] = -1
    
    for l in infh:
        try:
            line = l.decode()
        except AttributeError:
            line = l
        if (line_marker == "nomarker") and (tags_marker not in line): #read every line except with the tag marker
            pass
        elif (not line.startswith(line_marker)):
            continue
        
        if header_line is None and tags_marker in line:
            valid_tags, indices = _check_tags_ndx_line(line, tags)
            assert len(valid_tags) == len(tags), f"Valid tags: {valid_tags} doesn't match specified tags {tags}!"
            if verbose >=2:
                print("Reading output for %d valid tags: %s"%(len(valid_tags),", ".join(valid_tags)))
            if verbose >=2:
                print("indices:", indices)
            continue
        if ignore_incomplete and len(valid_tags) < len(tags):
            if verbose>=2:
                print("Ignore lines with imcomplete tags")
            continue
        if verbose >=4:
            print(line)
        reval = _line_parser(line,indices,dtypes=dtypes)
        if reval:
            if verbose >=4:
                print(reval)
            values.append(reval)
    infh.close()

    return valid_tags, values

def read_log(fn, tags, dtypes, line_marker = "SCORE:" , tags_marker = "description", header_line=None, ignore_incomplete=False, verbose=False):

    if not os.path.exists(fn):
        raise IOError("%s doesn't exist!"%fn)

    values = []
    if ".gz" in fn:
        infn = gzip.open(fn, 'rt')
    else:
        infn = open(fn, 'r')
    return read_log_fstream(infn, tags, dtypes, line_marker, tags_marker, header_line, ignore_incomplete, verbose)


def read_score_line(fn,tags, verbose=False):

    if not os.path.exists(fn):
        raise IOError("%s doesn't exist!"%fn)

    valid_tags, values = read_log(fn, tags,
                                   line_marker="SCORE:",
                                   tags_marker="description",
                                   verbose=verbose)

    return valid_tags, values

