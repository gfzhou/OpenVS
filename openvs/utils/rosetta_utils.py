'''Parse rosetta log files.'''

from __future__ import print_function
from io import BytesIO, TextIOWrapper
import os,sys
from typing import Iterable, Literal, Optional, Union
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

def _check_tags_ndx_line(line:str, tags:Iterable[str]) -> tuple[list[str], list[int]]:
    '''Read and split a line;find out tags' indices in it.
    
    Params
    ======
    - line: A string in a stream.
    - tags: A group of tagnames.

    Returns
    =======
    A list of vaild tag and a list of their indices.
    '''
    indices:list[int] = []
    valid_tags:list[str] = []

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

def _line_parser(line:str, indices:Optional[list[int]] = None,dtypes:tuple[type]=(float,) ) -> Union[list[object],Literal[False],str]:
    '''Parse fields in one line.

    Params
    ======
    - line: A line in a stream.
    - indices = None: Indices to locate in the split line.If `None`, return the original line.
    - dtypes = (float,): A group of date types to convert str into.When length of it is 1,all fields are parsed as the same date type in dtypes.
    
    Returns
    =======
    - False: When the line says the its result fails; or some field parse fail in the line.
    - A list of parsed value.
    '''
    values:list[object] = []
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
                dtype = dtypes[indices.index(i)]  # FIXME:is it right?
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


def read_log_fstream(infh: Union[TextIOWrapper, BytesIO],
                     tags: list[str],
                     dtypes: tuple[type],
                     line_marker: str = "SCORE:",
                     tags_marker: str = "description",
                     header_line: Optional[str] = None,
                     ignore_incomplete: bool = False,
                     verbose: int = False) -> tuple[list[str], list[Union[list[object],str]]]:
    '''Read log file stream and parse the lines into values.
    
    Params
    ======
    - infh: A file handle or any str/bytes stream.
    - tags: A list of tag names.
    - dtypes: A group of date types,corresponding to tags.
    - line_marker = 'SCORE:': Parse the lines if they start with line_marker.Parse all lines when line_marker is `nomarker`.
    - tags_marker = 'description': Lines containing it is parsed as header line.
    - header_line = None: A string contains tags.Supposed to be the header line of a file.If None,use `tags_marker` to find header line.
    - ignore_incomplete = False: Whether to skip incomplete parsed lines.
    - verbose = False: Log output level.Output more when it is higher.

    Returns
    =======
    1. A list of vaild tag names,
    2. and a list of parsed value.
    '''
    valid_tags:list[str] = []
    values:list[Union[list[object],str]] = []
    indices = []
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

def read_log(fn:str, tags:list[str], dtypes:tuple[type], line_marker:str = "SCORE:" , tags_marker:str = "description", header_line:Optional[str]=None, ignore_incomplete: bool = False, verbose: bool = False) -> tuple[list[str], list[Union[list[object],str]]]:
    '''Open a log file and parse the lines into values.
    
    Params
    ======
    - fn: Filename of log.
    - tags: A list of tag names.
    - dtypes: A group of date types,corresponding to tags.
    - line_marker = 'SCORE:': Parse the lines if they start with line_marker.Parse all lines when line_marker is `nomarker`.
    - tags_marker = 'description': Lines containing it is parsed as header line.
    - header_line = None: A string contains tags.Supposed to be the header line of a file.If None,use `tags_marker` to find header line.
    - ignore_incomplete = False: Whether to skip incomplete parsed lines.
    - verbose = False: Log output level.Output more when it is higher.

    Returns
    =======
    1. A list of vaild tag names,
    2. and a list of parsed value.
    '''
    if not os.path.exists(fn):
        raise IOError("%s doesn't exist!"%fn)

    values = []
    if ".gz" in fn:
        infn = gzip.open(fn, 'rt')
    else:
        infn = open(fn, 'r')
    return read_log_fstream(infn, tags, dtypes, line_marker, tags_marker, header_line, ignore_incomplete, verbose)


def read_score_line(fn:str,tags:list[str], verbose:bool=False) -> tuple[list[str], list[Union[list[object],str]]]:
    '''Read score lines of a log file and parse into values.
    
    Params
    ======
    - fn: Filename of log.
    - tags: A list of tag names.
    - verbose = False: Log output level.Output more when it is higher.

    Returns
    =======
    1. A list of vaild tag names,
    2. and a list of parsed value.
    '''
    if not os.path.exists(fn):
        raise IOError("%s doesn't exist!"%fn)

    valid_tags, values = read_log(fn, tags,
                                   line_marker="SCORE:",
                                   tags_marker="description",
                                   verbose=verbose)

    return valid_tags, values
