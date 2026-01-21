# -*- coding: utf-8 -*-
"""
Flask API for Nephrotic Syndrome Genetics Dashboard - FIXED SIGNIFICANCE CRITERIA
- Significance requires: p <= 0.01 (Bonferroni) AND AF >= 0.05 (either pop) AND FST >= 0.15
- All three conditions must be met in the SAME population comparison
"""

from flask import Flask, jsonify, request, render_template, redirect, url_for
from flask_cors import CORS
import sqlite3
from typing import Dict, List
import json
import os
import re
import numpy as np
from scipy import stats
from datetime import datetime
import time

app = Flask(__name__)
CORS(app)

# Significance thresholds
BONFERRONI_P_THRESHOLD = 0.01
AF_THRESHOLD = 0.05
FST_THRESHOLD = 0.15

# Cache for statistics (to avoid expensive queries)
_stats_cache = None
_stats_cache_time = None
CACHE_DURATION = 3600  # Cache for 1 hour (in seconds)

# Database path
def get_database_path():
    parent_db = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nephrotic_syndrome.db')
    if os.path.exists(parent_db):
        return parent_db
    
    current_db = 'nephrotic_syndrome.db'
    if os.path.exists(current_db):
        return current_db
    
    up_one = '../nephrotic_syndrome.db'
    if os.path.exists(up_one):
        return up_one
    
    return current_db

DATABASE_PATH = get_database_path()

def get_db_connection():
    if not os.path.exists(DATABASE_PATH):
        raise FileNotFoundError(f"Database not found at: {os.path.abspath(DATABASE_PATH)}")
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA cache_size = 10000")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def create_indexes_if_needed():
    """Create database indexes to speed up queries (runs once)"""
    conn = get_db_connection()
    try:
        # Check if indexes already exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_st_variant_pop'")
        if cursor.fetchone() is None:
            print("[INFO] Creating database indexes for faster queries...")
            
            # Index for statistical_tests
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_st_variant_pop 
                ON statistical_tests(variant_id, population)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_st_pvalue_fst 
                ON statistical_tests(chi_square_p_value, fst_value)
            """)
            
            # Index for allele_frequencies
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_af_variant_pop 
                ON allele_frequencies(variant_id, population)
            """)
            
            # Index for variants
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_variants_gene 
                ON variants(gene_name)
            """)
            
            conn.commit()
            print("[OK] Indexes created successfully")
        else:
            print("[OK] Database indexes already exist")
    except Exception as e:
        print(f"[WARNING] Could not create indexes: {e}")
    finally:
        conn.close()


def dict_from_row(row):
    """Convert sqlite3.Row to dictionary"""
    return {key: row[key] for key in row.keys()}


def check_variant_significance(variant_id, conn):
    """
    Check if variant meets NEW significance criteria:
    - At least ONE population where ALL THREE conditions are met:
      1. p-value ≤ 0.01 (Bonferroni)
      2. AF ≥ 0.05 in either EUR or that population  
      3. FST ≥ 0.15
    
    Returns: (is_significant, significant_populations)
    """
    cursor = conn.cursor()
    
    # Get statistical tests for this variant
    cursor.execute("""
        SELECT st.population, st.chi_square_p_value, st.fst_value,
               af_eur.allele_frequency as af_eur,
               af_pop.allele_frequency as af_pop
        FROM statistical_tests st
        LEFT JOIN allele_frequencies af_eur 
            ON st.variant_id = af_eur.variant_id AND af_eur.population = 'EUR'
        LEFT JOIN allele_frequencies af_pop 
            ON st.variant_id = af_pop.variant_id AND af_pop.population = st.population
        WHERE st.variant_id = ?
    """, (variant_id,))
    
    significant_pops = []
    
    for row in cursor.fetchall():
        pop = row['population']
        p_value = row['chi_square_p_value']
        fst_value = row['fst_value']
        af_eur = row['af_eur'] or 0
        af_pop = row['af_pop'] or 0
        
        # Check all three conditions for this population
        p_significant = p_value is not None and p_value <= BONFERRONI_P_THRESHOLD
        af_sufficient = (af_eur >= AF_THRESHOLD) or (af_pop >= AF_THRESHOLD)
        fst_significant = fst_value is not None and fst_value >= FST_THRESHOLD
        
        # All three must be true
        if p_significant and af_sufficient and fst_significant:
            significant_pops.append({
                'population': pop,
                'p_value': p_value,
                'fst_value': fst_value,
                'af_eur': af_eur,
                'af_pop': af_pop,
                'meets_all_criteria': True
            })
    
    return len(significant_pops) > 0, significant_pops


def format_pubmed_citations(paper_string):
    """Convert paper citations to structured format with links"""
    if not paper_string or paper_string is None or str(paper_string).strip() == '':
        return []
    
    citations = []
    parts = str(paper_string).split(',')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if part.upper().startswith('PMC'):
            citations.append({
                'id': part,
                'type': 'PMC',
                'url': f'https://www.ncbi.nlm.nih.gov/pmc/articles/{part}/'
            })
        elif part.isdigit():
            citations.append({
                'id': part,
                'type': 'PMID',
                'url': f'https://pubmed.ncbi.nlm.nih.gov/{part}/'
            })
        elif part.startswith('http'):
            citations.append({
                'id': part.split('/')[-1] or 'Link',
                'type': 'URL',
                'url': part
            })
        else:
            pmid_match = re.search(r'\d{7,8}', part)
            if pmid_match:
                pmid = pmid_match.group()
                citations.append({
                    'id': pmid,
                    'type': 'PMID',
                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
                })
    
    return citations


@app.route('/')
def index():
    """Landing page"""
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard interface"""
    return render_template('index.html')


@app.route('/api/genes', methods=['GET'])
def get_genes():
    """Returns list of all genes with basic information"""
    conn = get_db_connection()
    
    cursor = conn.execute("""
        SELECT 
            g.gene_name,
            g.ensembl_id,
            g.chromosome,
            g.srns_ssns,
            COALESCE(v.variant_count, 0) as variant_count
        FROM genes g
        LEFT JOIN (
            SELECT gene_name, COUNT(*) as variant_count
            FROM variants
            GROUP BY gene_name
        ) v ON g.gene_name = v.gene_name
        ORDER BY g.gene_name
    """)
    
    genes = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({
        'success': True,
        'count': len(genes),
        'data': genes
    })


@app.route('/api/gene/<gene_name>', methods=['GET'])
def get_gene_details(gene_name):
    """Returns complete information for a specific gene with formatted citations"""
    conn = get_db_connection()
    
    gene_query = """
        SELECT 
            g.*,
            gl.omim_link,
            gl.gwas_link,
            gl.clinpgx_link,
            gl.malacards_link,
            gl.function_link,
            gl.pathway_link,
            gl.source_paper_1,
            gl.source_paper_2,
            gl.source_paper_3
        FROM genes g
        LEFT JOIN gene_links gl ON g.gene_name = gl.gene_name
        WHERE g.gene_name = ?
    """
    cursor = conn.execute(gene_query, (gene_name,))
    gene = cursor.fetchone()
    
    if not gene:
        conn.close()
        return jsonify({'success': False, 'error': 'Gene not found'}), 404
    
    gene_data = dict_from_row(gene)
    
    citations = []
    for i in range(1, 4):
        paper_field = f'source_paper_{i}'
        if gene_data.get(paper_field):
            formatted = format_pubmed_citations(gene_data[paper_field])
            citations.extend(formatted)
    
    gene_data['citations'] = citations
    gene_data['citation_count'] = len(citations)
    
    cursor = conn.execute("""
        SELECT COUNT(*) as count FROM variants WHERE gene_name = ?
    """, (gene_name,))
    gene_data['variant_count'] = cursor.fetchone()['count']
    
    # Count variants meeting NEW significance criteria
    cursor = conn.execute("""
        SELECT variant_id FROM variants WHERE gene_name = ?
    """, (gene_name,))
    
    sig_count = 0
    for row in cursor.fetchall():
        is_sig, _ = check_variant_significance(row['variant_id'], conn)
        if is_sig:
            sig_count += 1
    
    gene_data['significant_variant_count'] = sig_count
    
    conn.close()
    
    return jsonify({
        'success': True,
        'data': gene_data
    })


@app.route('/api/gene/<gene_name>/variants', methods=['GET'])
def get_gene_variants(gene_name):
    """Returns variants for a specific gene with NEW significance criteria"""
    population = request.args.get('population', 'all')
    significant_only = request.args.get('significant', 'false').lower() == 'true'
    after_ld = request.args.get('after_ld', 'false').lower() == 'true'
    
    conn = get_db_connection()
    
    query = """
        SELECT DISTINCT
            v.variant_id,
            v.rs_id,
            v.chrom,
            v.position,
            v.ref_allele,
            v.alt_allele,
            v.global_af,
            v.dbsnp_link
        FROM variants v
        WHERE v.gene_name = ?
    """
    params = [gene_name]
    
    if after_ld:
        query += """
            AND EXISTS (
                SELECT 1 FROM ld_status ld
                WHERE ld.variant_id = v.variant_id
                AND ld.ld_pruned = 1
            )
        """
    
    query += " ORDER BY v.chrom, v.position"
    
    cursor = conn.execute(query, params)
    variants = [dict_from_row(row) for row in cursor.fetchall()]
    
    if len(variants) == 0:
        conn.close()
        return jsonify({
            'success': True,
            'gene_name': gene_name,
            'count': 0,
            'data': []
        })
    
    variant_ids = [v['variant_id'] for v in variants]
    placeholders = ','.join(['?'] * len(variant_ids))
    
    # Get allele frequencies
    cursor = conn.execute(f"""
        SELECT variant_id, population, allele_frequency
        FROM allele_frequencies
        WHERE variant_id IN ({placeholders})
    """, variant_ids)
    
    af_by_variant = {}
    for row in cursor.fetchall():
        vid = row['variant_id']
        if vid not in af_by_variant:
            af_by_variant[vid] = {}
        af_by_variant[vid][row['population']] = row['allele_frequency']
    
    # Get statistical tests
    cursor = conn.execute(f"""
        SELECT variant_id, population, fst_value, chi_square_p_value,
               is_significant_chisq, is_significant_bonferroni, is_significant_fst
        FROM statistical_tests
        WHERE variant_id IN ({placeholders})
    """, variant_ids)
    
    stats_by_variant = {}
    for row in cursor.fetchall():
        vid = row['variant_id']
        if vid not in stats_by_variant:
            stats_by_variant[vid] = []
        stats_by_variant[vid].append(dict_from_row(row))
    
    # Get LD status
    cursor = conn.execute(f"""
        SELECT variant_id, ld_pruned
        FROM ld_status
        WHERE variant_id IN ({placeholders})
    """, variant_ids)
    
    ld_by_variant = {row['variant_id']: row['ld_pruned'] for row in cursor.fetchall()}
    
    # Process each variant with NEW significance criteria
    filtered_variants = []
    
    for variant in variants:
        vid = variant['variant_id']
        variant['allele_frequencies'] = af_by_variant.get(vid, {})
        variant['statistics'] = stats_by_variant.get(vid, [])
        variant['in_ld'] = 'Yes' if ld_by_variant.get(vid, 0) else 'No'
        
        # Check NEW significance criteria
        is_sig, sig_pops = check_variant_significance(vid, conn)
        variant['is_significant'] = is_sig
        variant['significant_populations'] = sig_pops
        
        # Apply filter if requested
        if significant_only and not is_sig:
            continue
            
        filtered_variants.append(variant)
    
    conn.close()
    
    return jsonify({
        'success': True,
        'gene_name': gene_name,
        'count': len(filtered_variants),
        'criteria': {
            'p_threshold': BONFERRONI_P_THRESHOLD,
            'af_threshold': AF_THRESHOLD,
            'fst_threshold': FST_THRESHOLD,
            'description': 'p ≤ 0.01 (Bonferroni) AND AF ≥ 0.05 (either pop) AND FST ≥ 0.15'
        },
        'data': filtered_variants
    })


@app.route('/api/variant/<rs_id>', methods=['GET'])
def get_variant_details(rs_id):
    """Returns complete information for a specific variant"""
    conn = get_db_connection()
    
    cursor = conn.execute("""
        SELECT 
            v.*,
            g.gene_name,
            g.ensembl_id,
            g.srns_ssns,
            gl.function_link,
            gl.pathway_link
        FROM variants v
        JOIN genes g ON v.gene_name = g.gene_name
        LEFT JOIN gene_links gl ON g.gene_name = gl.gene_name
        WHERE v.rs_id = ?
    """, (rs_id,))
    
    variant = cursor.fetchone()
    
    if not variant:
        conn.close()
        return jsonify({'success': False, 'error': 'Variant not found'}), 404
    
    variant_data = dict_from_row(variant)
    variant_id = variant_data['variant_id']
    
    cursor = conn.execute("""
        SELECT population, allele_frequency
        FROM allele_frequencies
        WHERE variant_id = ?
        ORDER BY population
    """, (variant_id,))
    variant_data['allele_frequencies'] = {
        row['population']: row['allele_frequency']
        for row in cursor.fetchall()
    }
    
    cursor = conn.execute("""
        SELECT *
        FROM statistical_tests
        WHERE variant_id = ?
        ORDER BY population
    """, (variant_id,))
    variant_data['statistics'] = [dict_from_row(row) for row in cursor.fetchall()]
    
    cursor = conn.execute("""
        SELECT population, in_ld, ld_pruned
        FROM ld_status
        WHERE variant_id = ?
    """, (variant_id,))
    variant_data['ld_status'] = [dict_from_row(row) for row in cursor.fetchall()]
    
    # Check NEW significance
    is_sig, sig_pops = check_variant_significance(variant_id, conn)
    variant_data['is_significant'] = is_sig
    variant_data['significant_populations'] = sig_pops
    
    conn.close()
    
    return jsonify({
        'success': True,
        'data': variant_data
    })


@app.route('/api/variant/<rs_id>/calculate-chisquare', methods=['POST'])
def calculate_chisquare_runtime(rs_id):
    """Calculate chi-square test between any two populations for a variant"""
    try:
        data = request.get_json()
        pop1 = data.get('population1')
        pop2 = data.get('population2')
        
        if not pop1 or not pop2:
            return jsonify({
                'success': False, 
                'error': 'Both population1 and population2 are required'
            }), 400
        
        if pop1 == pop2:
            return jsonify({
                'success': False,
                'error': 'Please select two different populations'
            }), 400
        
        conn = get_db_connection()
        
        cursor = conn.execute(
            "SELECT variant_id FROM variants WHERE rs_id = ?",
            (rs_id,)
        )
        variant = cursor.fetchone()
        
        if not variant:
            conn.close()
            return jsonify({'success': False, 'error': 'Variant not found'}), 404
        
        variant_id = variant['variant_id']
        
        cursor = conn.execute("""
            SELECT population, allele_frequency
            FROM allele_frequencies
            WHERE variant_id = ? AND population IN (?, ?)
        """, (variant_id, pop1, pop2))
        
        frequencies = {row['population']: row['allele_frequency'] for row in cursor.fetchall()}
        conn.close()
        
        if pop1 not in frequencies or pop2 not in frequencies:
            return jsonify({
                'success': False,
                'error': f'Allele frequency data not available for one or both populations'
            }), 404
        
        af1 = frequencies[pop1]
        af2 = frequencies[pop2]
       
        epsilon = 1e-10
        af1_safe = max(af1, epsilon)
        af2_safe = max(af2, epsilon)
        
        chi_sq_1 = ((af1 - af2) ** 2) / af2_safe
        chi_sq_2 = ((af2 - af1) ** 2) / af1_safe
        chi_sq_total = chi_sq_1 + chi_sq_2
        
        p_value = float(1 - stats.chi2.cdf(chi_sq_total, df=1))
        
        is_significant_05 = p_value <= 0.05
        is_significant_01 = p_value <= 0.01
        
        af_difference = abs(af1 - af2)
        
        return jsonify({
            'success': True,
            'data': {
                'rs_id': rs_id,
                'population1': pop1,
                'population2': pop2,
                'af1': af1,
                'af2': af2,
                'af_difference': af_difference,
                'chi_square_1': float(chi_sq_1),
                'chi_square_2': float(chi_sq_2),
                'chi_square_total': float(chi_sq_total),
                'p_value': p_value,
                'is_significant_p05': is_significant_05,
                'is_significant_p01': is_significant_01,
                'interpretation': {
                    'significance': 'Significant' if is_significant_05 else 'Not significant',
                    'effect_size': 'Large' if af_difference > 0.2 else 'Medium' if af_difference > 0.1 else 'Small'
                }
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/search', methods=['GET'])
def search():
    """Search for genes or variants"""
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({'success': False, 'error': 'No search query provided'}), 400
    
    conn = get_db_connection()
    
    exact_match_cursor = conn.execute("""
        SELECT gene_name, ensembl_id, chromosome, srns_ssns
        FROM genes
        WHERE ensembl_id = ?
        LIMIT 1
    """, (query,))
    
    exact_match = exact_match_cursor.fetchone()
    
    if exact_match:
        gene_data = dict_from_row(exact_match)
        conn.close()
        return jsonify({
            'success': True,
            'query': query,
            'redirect': True,
            'redirect_type': 'gene',
            'redirect_to': gene_data['gene_name'],
            'results': {
                'genes': [gene_data],
                'variants': []
            }
        })
    
    gene_cursor = conn.execute("""
        SELECT gene_name, ensembl_id, chromosome, srns_ssns
        FROM genes
        WHERE gene_name LIKE ? OR ensembl_id LIKE ?
        ORDER BY 
            CASE 
                WHEN gene_name = ? THEN 1
                WHEN ensembl_id = ? THEN 2
                WHEN gene_name LIKE ? THEN 3
                WHEN ensembl_id LIKE ? THEN 4
                ELSE 5
            END,
            gene_name
        LIMIT 50
    """, (f'%{query}%', f'%{query}%', query, query, f'{query}%', f'{query}%'))
    
    genes = [dict_from_row(row) for row in gene_cursor.fetchall()]
    
    variant_cursor = conn.execute("""
        SELECT v.rs_id, v.chrom, v.position, v.gene_name
        FROM variants v
        WHERE v.rs_id LIKE ? OR v.gene_name LIKE ?
        ORDER BY
            CASE
                WHEN v.rs_id = ? THEN 1
                WHEN v.rs_id LIKE ? THEN 2
                ELSE 3
            END,
            v.rs_id
        LIMIT 50
    """, (f'%{query}%', f'%{query}%', query, f'{query}%'))
    
    variants = [dict_from_row(row) for row in variant_cursor.fetchall()]
    
    conn.close()
    
    return jsonify({
        'success': True,
        'query': query,
        'results': {
            'genes': genes,
            'variants': variants
        }
    })


@app.route('/api/variants', methods=['GET'])
def get_all_variants():
    """Returns all variants with pagination support"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    significant_only = request.args.get('significant', 'false').lower() == 'true'
    
    # Limit per_page to prevent memory issues
    per_page = min(per_page, 100)
    offset = (page - 1) * per_page
    
    conn = get_db_connection()
    
    if significant_only:
        # Get significant variants using the same criteria
        count_query = """
            SELECT COUNT(DISTINCT v.variant_id) as count
            FROM variants v
            WHERE EXISTS (
                SELECT 1
                FROM statistical_tests st
                LEFT JOIN allele_frequencies af_eur 
                    ON st.variant_id = af_eur.variant_id 
                    AND af_eur.population = 'EUR'
                LEFT JOIN allele_frequencies af_pop 
                    ON st.variant_id = af_pop.variant_id 
                    AND af_pop.population = st.population
                WHERE st.variant_id = v.variant_id
                    AND st.chi_square_p_value <= ?
                    AND st.fst_value >= ?
                    AND (
                        COALESCE(af_eur.allele_frequency, 0) >= ? 
                        OR COALESCE(af_pop.allele_frequency, 0) >= ?
                    )
            )
        """
        cursor = conn.execute(count_query, (BONFERRONI_P_THRESHOLD, FST_THRESHOLD, AF_THRESHOLD, AF_THRESHOLD))
        total_count = cursor.fetchone()['count']
        
        variants_query = """
            SELECT DISTINCT v.variant_id, v.rs_id, v.chrom, v.position, 
                   v.ref_allele, v.alt_allele, v.global_af, v.dbsnp_link, v.gene_name
            FROM variants v
            WHERE EXISTS (
                SELECT 1
                FROM statistical_tests st
                LEFT JOIN allele_frequencies af_eur 
                    ON st.variant_id = af_eur.variant_id 
                    AND af_eur.population = 'EUR'
                LEFT JOIN allele_frequencies af_pop 
                    ON st.variant_id = af_pop.variant_id 
                    AND af_pop.population = st.population
                WHERE st.variant_id = v.variant_id
                    AND st.chi_square_p_value <= ?
                    AND st.fst_value >= ?
                    AND (
                        COALESCE(af_eur.allele_frequency, 0) >= ? 
                        OR COALESCE(af_pop.allele_frequency, 0) >= ?
                    )
            )
            ORDER BY v.chrom, v.position
            LIMIT ? OFFSET ?
        """
        cursor = conn.execute(variants_query, (BONFERRONI_P_THRESHOLD, FST_THRESHOLD, AF_THRESHOLD, AF_THRESHOLD, per_page, offset))
    else:
        # Get total count
        cursor = conn.execute("SELECT COUNT(*) as count FROM variants")
        total_count = cursor.fetchone()['count']
        
        # Get paginated variants
        cursor = conn.execute("""
            SELECT variant_id, rs_id, chrom, position, ref_allele, alt_allele, 
                   global_af, dbsnp_link, gene_name
            FROM variants
            ORDER BY chrom, position
            LIMIT ? OFFSET ?
        """, (per_page, offset))
    
    variants = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    
    total_pages = (total_count + per_page - 1) // per_page
    
    return jsonify({
        'success': True,
        'data': variants,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total_count': total_count,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
    })


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Returns overall database statistics with NEW significance criteria - CACHED"""
    global _stats_cache, _stats_cache_time
    
    # Check if we should force refresh
    force_refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    # Check if cache is valid
    current_time = time.time()
    cache_is_valid = (
        _stats_cache is not None and 
        _stats_cache_time is not None and 
        (current_time - _stats_cache_time) < CACHE_DURATION and
        not force_refresh
    )
    
    if cache_is_valid:
        # Return cached data
        return jsonify({
            'success': True,
            'data': _stats_cache,
            'cached': True,
            'cache_age_seconds': int(current_time - _stats_cache_time)
        })
    
    # Cache miss or expired - calculate statistics
    conn = get_db_connection()
    
    stats = {}
    
    # Total genes
    cursor = conn.execute("SELECT COUNT(DISTINCT gene_name) as count FROM genes")
    stats['total_genes'] = cursor.fetchone()['count']
    
    # Total variants
    cursor = conn.execute("SELECT COUNT(*) as count FROM variants")
    stats['total_variants'] = cursor.fetchone()['count']
    
    # Count significant variants - OPTIMIZED using JOINs instead of EXISTS
    # A variant is significant if ANY population meets ALL THREE criteria:
    # 1. p-value <= 0.01, 2. AF >= 0.05 in either EUR or that population, 3. FST >= 0.15
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT st.variant_id) as count
        FROM statistical_tests st
        INNER JOIN allele_frequencies af_eur 
            ON st.variant_id = af_eur.variant_id 
            AND af_eur.population = 'EUR'
        INNER JOIN allele_frequencies af_pop 
            ON st.variant_id = af_pop.variant_id 
            AND af_pop.population = st.population
        WHERE st.chi_square_p_value <= ?
            AND st.fst_value >= ?
            AND (af_eur.allele_frequency >= ? OR af_pop.allele_frequency >= ?)
    """, (BONFERRONI_P_THRESHOLD, FST_THRESHOLD, AF_THRESHOLD, AF_THRESHOLD))
    
    stats['significant_variants'] = cursor.fetchone()['count']
    
    # Variants per gene distribution
    cursor = conn.execute("""
        SELECT 
            MIN(variant_count) as min_variants,
            MAX(variant_count) as max_variants,
            AVG(variant_count) as avg_variants
        FROM (
            SELECT gene_name, COUNT(*) as variant_count
            FROM variants
            GROUP BY gene_name
        )
    """)
    dist = cursor.fetchone()
    stats['variants_per_gene'] = {
        'min': dist['min_variants'],
        'max': dist['max_variants'],
        'avg': round(dist['avg_variants'], 2)
    }
    
    stats['significance_criteria'] = {
        'p_threshold': BONFERRONI_P_THRESHOLD,
        'af_threshold': AF_THRESHOLD,
        'fst_threshold': FST_THRESHOLD,
        'description': 'p <= 0.01 (Bonferroni) AND AF >= 0.05 (either pop) AND FST >= 0.15'
    }
    
    conn.close()
    
    # Update cache
    _stats_cache = stats
    _stats_cache_time = current_time
    
    return jsonify({
        'success': True,
        'data': stats,
        'cached': False
    })


@app.route('/api/variants/significant-ld', methods=['GET'])
def get_significant_ld_variants():
    """Returns significant variants that passed LD pruning with allele frequencies"""
    conn = get_db_connection()
    gene_name = request.args.get('gene')
    
    # Get variants that are significant AND kept after LD pruning
    query = """
        SELECT DISTINCT v.variant_id, v.rs_id, v.gene_name, v.chrom, v.position
        FROM variants v
        INNER JOIN ld_status ld ON v.variant_id = ld.variant_id AND ld.ld_pruned = 1
        INNER JOIN statistical_tests st ON v.variant_id = st.variant_id
        INNER JOIN allele_frequencies af_eur ON v.variant_id = af_eur.variant_id AND af_eur.population = 'EUR'
        INNER JOIN allele_frequencies af_pop ON v.variant_id = af_pop.variant_id AND af_pop.population = st.population
        WHERE st.chi_square_p_value <= ?
            AND st.fst_value >= ?
            AND (af_eur.allele_frequency >= ? OR af_pop.allele_frequency >= ?)
    """
    params = [BONFERRONI_P_THRESHOLD, FST_THRESHOLD, AF_THRESHOLD, AF_THRESHOLD]
    
    if gene_name:
        query += " AND v.gene_name = ?"
        params.append(gene_name)
    
    query += " LIMIT 50"
    cursor = conn.execute(query, params)
    
    variants = []
    variant_ids = []
    
    for row in cursor.fetchall():
        variant_ids.append(row['variant_id'])
        variants.append({
            'variant_id': row['variant_id'],
            'rs_id': row['rs_id'],
            'gene_name': row['gene_name'],
            'chrom': row['chrom'],
            'position': row['position']
        })
    
    # Get allele frequencies for all populations for these variants
    if variant_ids:
        placeholders = ','.join(['?' for _ in variant_ids])
        cursor = conn.execute(f"""
            SELECT variant_id, population, allele_frequency
            FROM allele_frequencies
            WHERE variant_id IN ({placeholders})
        """, variant_ids)
        
        # Build frequency map
        freq_map = {}
        for row in cursor.fetchall():
            vid = row['variant_id']
            if vid not in freq_map:
                freq_map[vid] = {}
            freq_map[vid][row['population']] = row['allele_frequency']
        
        # Add frequencies to variants
        for variant in variants:
            variant['frequencies'] = freq_map.get(variant['variant_id'], {})
    
    conn.close()
    
    return jsonify({
        'success': True,
        'data': variants,
        'count': len(variants)
    })


@app.route('/api/statistics/clear-cache', methods=['POST'])
def clear_statistics_cache():
    """Manually clear the statistics cache"""
    global _stats_cache, _stats_cache_time
    _stats_cache = None
    _stats_cache_time = None
    return jsonify({
        'success': True,
        'message': 'Statistics cache cleared'
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("Nephrotic Syndrome Genetics Dashboard API - FIXED SIGNIFICANCE")
    print("=" * 70)
    print(f"\nNEW Significance Criteria (ALL THREE must be met):")
    print(f"   1. p-value <= {BONFERRONI_P_THRESHOLD} (Bonferroni)")
    print(f"   2. AF >= {AF_THRESHOLD} in either EUR or comparison population")
    print(f"   3. FST >= {FST_THRESHOLD}")
    print(f"\nDatabase location: {os.path.abspath(DATABASE_PATH)}")
    
    try:
        conn = get_db_connection()
        cursor = conn.execute("SELECT COUNT(*) as count FROM genes")
        gene_count = cursor.fetchone()['count']
        cursor = conn.execute("SELECT COUNT(*) as count FROM variants")
        variant_count = cursor.fetchone()['count']
        conn.close()
        print(f"[OK] Database connected successfully")
        print(f"  - {gene_count} genes")
        print(f"  - {variant_count} variants")
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        exit(1)
    
    # Create indexes if they don't exist (speeds up queries dramatically)
    create_indexes_if_needed()
    
    # Pre-warm the statistics cache on startup
    print("\n[INFO] Pre-warming statistics cache (this may take a few seconds)...")
    try:
        warm_start = time.time()
        conn = get_db_connection()
        
        warm_stats = {}
        
        # Total genes
        cursor = conn.execute("SELECT COUNT(DISTINCT gene_name) as count FROM genes")
        warm_stats['total_genes'] = cursor.fetchone()['count']
        
        # Total variants
        cursor = conn.execute("SELECT COUNT(*) as count FROM variants")
        warm_stats['total_variants'] = cursor.fetchone()['count']
        
        # Count significant variants - OPTIMIZED using JOINs instead of EXISTS
        # A variant is significant if ANY population meets ALL THREE criteria:
        # 1. p-value <= 0.01, 2. AF >= 0.05 in either EUR or that population, 3. FST >= 0.15
        cursor = conn.execute("""
            SELECT COUNT(DISTINCT st.variant_id) as count
            FROM statistical_tests st
            INNER JOIN allele_frequencies af_eur 
                ON st.variant_id = af_eur.variant_id 
                AND af_eur.population = 'EUR'
            INNER JOIN allele_frequencies af_pop 
                ON st.variant_id = af_pop.variant_id 
                AND af_pop.population = st.population
            WHERE st.chi_square_p_value <= ?
                AND st.fst_value >= ?
                AND (af_eur.allele_frequency >= ? OR af_pop.allele_frequency >= ?)
        """, (BONFERRONI_P_THRESHOLD, FST_THRESHOLD, AF_THRESHOLD, AF_THRESHOLD))
        warm_stats['significant_variants'] = cursor.fetchone()['count']
        
        # Variants per gene distribution
        cursor = conn.execute("""
            SELECT 
                MIN(variant_count) as min_variants,
                MAX(variant_count) as max_variants,
                AVG(variant_count) as avg_variants
            FROM (
                SELECT gene_name, COUNT(*) as variant_count
                FROM variants
                GROUP BY gene_name
            )
        """)
        dist = cursor.fetchone()
        warm_stats['variants_per_gene'] = {
            'min': dist['min_variants'],
            'max': dist['max_variants'],
            'avg': round(dist['avg_variants'], 2)
        }
        
        warm_stats['significance_criteria'] = {
            'p_threshold': BONFERRONI_P_THRESHOLD,
            'af_threshold': AF_THRESHOLD,
            'fst_threshold': FST_THRESHOLD,
            'description': 'p <= 0.01 (Bonferroni) AND AF >= 0.05 (either pop) AND FST >= 0.15'
        }
        
        conn.close()
        
        # Store in cache - update module level variables using globals()
        globals()['_stats_cache'] = warm_stats
        globals()['_stats_cache_time'] = time.time()
        
        warm_duration = time.time() - warm_start
        print(f"[OK] Cache warmed in {warm_duration:.1f} seconds")
        print(f"  - Significant variants: {warm_stats['significant_variants']}")
    except Exception as e:
        print(f"[WARNING] Cache warm-up failed: {e}")
        print("  Statistics will be calculated on first request")
    
    print("\nStarting Flask server...")
    print("API available at: http://localhost:5000")
    print("\n" + "=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)