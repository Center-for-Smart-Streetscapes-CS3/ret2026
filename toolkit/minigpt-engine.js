/* Real char-level Mini-GPT (one causal transformer block) + char RNN.
   Runs entirely on CPU with flat Float32Arrays. Genuine forward + backprop + Adam.
   Registers window.MiniGPT. Teaching-scale: learns real character statistics; not GPT-2 quality. */
(function (global) {
  'use strict';

  function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; var t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
  function gelu(x){ return 0.5*x*(1+Math.tanh(0.7978845608*(x+0.044715*x*x*x))); }
  function dgelu(x){ var t=Math.tanh(0.7978845608*(x+0.044715*x*x*x)); var s=0.7978845608*(1+3*0.044715*x*x); return 0.5*(1+t)+0.5*x*(1-t*t)*s; }

  function MiniGPT(text, cfg){
    cfg = cfg || {};
    this.arch = cfg.arch || 'gpt';
    this.d = cfg.d || 48;            // model width
    this.h = cfg.h || 3;             // heads
    this.T = cfg.T || 32;            // context length
    this.hd = this.d / this.h;
    this.lr = cfg.lr || 0.0025;
    this.setText(text, cfg.units);
    this.rand = mulberry32(1234);
    this.step = 0;
    this.initParams();
  }

  MiniGPT.prototype.setText = function(text, units){
    var i;
    if (units && units.length){ // word / token level
      this.wordMode = true; this.joinSep = ' ';
      var seen={}, toks=[]; this.data=new Int32Array(units.length);
      for(i=0;i<units.length;i++){ var u=units[i]; if(seen[u]===undefined){ seen[u]=toks.length; toks.push(u); } this.data[i]=seen[u]; }
      this.chars=toks; this.stoi=seen; this.V=toks.length; this.text=units.join(' ');
    } else {
      this.wordMode = false; this.joinSep = '';
      this.text = text || '';
      var set = {}, chars = [];
      for (i=0;i<this.text.length;i++){ var c=this.text[i]; if(set[c]===undefined){ set[c]=chars.length; chars.push(c); } }
      this.chars = chars; this.stoi = set; this.V = chars.length;
      this.data = new Int32Array(this.text.length);
      for (i=0;i<this.text.length;i++) this.data[i]=set[this.text[i]];
    }
  };

  MiniGPT.prototype.rn = function(n,scale){ var a=new Float32Array(n); for(var i=0;i<n;i++){ // Box-Muller
      var u=0,v=0; while(!u)u=this.rand(); while(!v)v=this.rand(); a[i]=Math.sqrt(-2*Math.log(u))*Math.cos(6.2831853*v)*scale; } return a; };

  MiniGPT.prototype.initParams = function(){
    var d=this.d, V=this.V, T=this.T, ff=4*d;
    var s=1/Math.sqrt(d);
    this.p = {
      wte:this.rn(V*d,0.02), wpe:this.rn(T*d,0.02),
      ln1g:ones(d), ln1b:zeros(d),
      wq:this.rn(d*d,s), wk:this.rn(d*d,s), wv:this.rn(d*d,s), wo:this.rn(d*d,s),
      ln2g:ones(d), ln2b:zeros(d),
      wfc:this.rn(d*ff,s), bfc:zeros(ff), wproj:this.rn(ff*d,1/Math.sqrt(ff)), bproj:zeros(d),
      lnfg:ones(d), lnfb:zeros(d)
    };
    // RNN params (used when arch==='rnn')
    this.rp = { r_wxh:this.rn(d*d,s), r_whh:this.rn(d*d,s), r_bh:zeros(d), r_emb:this.rn(V*d,0.02) };
    this.g = {}; this.m = {}; this.v = {};
    var self=this; [this.p,this.rp].forEach(function(P){ Object.keys(P).forEach(function(k){ self.g[k]=new Float32Array(P[k].length); self.m[k]=new Float32Array(P[k].length); self.v[k]=new Float32Array(P[k].length); }); });
    function zeros(n){ return new Float32Array(n); }
    function ones(n){ var a=new Float32Array(n); a.fill(1); return a; }
  };

  function layernorm(x,t0,d,g,b,gi,bi){ // x is flat [T*d]; normalize row t0; returns {out,mean,inv,xhat}
    var mean=0,i,off=t0*d; for(i=0;i<d;i++)mean+=x[off+i]; mean/=d;
    var vv=0; for(i=0;i<d;i++){ var dx=x[off+i]-mean; vv+=dx*dx; } vv/=d;
    var inv=1/Math.sqrt(vv+1e-5); var out=new Float32Array(d), xhat=new Float32Array(d);
    for(i=0;i<d;i++){ xhat[i]=(x[off+i]-mean)*inv; out[i]=xhat[i]*g[gi?gi:0+i]; }
    return {mean:mean,inv:inv,xhat:xhat};
  }

  // forward one window starting at position `start`; store cache; return per-position hf + loss pieces
  MiniGPT.prototype.forwardGPT = function(ids, targets, wantGrad){
    var d=this.d,h=this.h,hd=this.hd,T=ids.length,V=this.V,ff=4*d,p=this.p,i,j,t,s,hh;
    var x0=new Float32Array(T*d);
    for(t=0;t<T;t++){ var id=ids[t]; for(i=0;i<d;i++) x0[t*d+i]=p.wte[id*d+i]+p.wpe[t*d+i]; }
    // ln1
    var ln1={mean:new Float32Array(T),inv:new Float32Array(T),xhat:new Float32Array(T*d)}, h1=new Float32Array(T*d);
    for(t=0;t<T;t++){ var m0=0; for(i=0;i<d;i++)m0+=x0[t*d+i]; m0/=d; var vv=0; for(i=0;i<d;i++){var dv=x0[t*d+i]-m0; vv+=dv*dv;} vv/=d; var inv=1/Math.sqrt(vv+1e-5); ln1.mean[t]=m0; ln1.inv[t]=inv; for(i=0;i<d;i++){ var xh=(x0[t*d+i]-m0)*inv; ln1.xhat[t*d+i]=xh; h1[t*d+i]=xh*p.ln1g[i]+p.ln1b[i]; } }
    // q,k,v = h1 @ W  (W is [d,d], row-major out index o: sum_i h1[i]*W[o*d+i])
    var q=matmul(h1,p.wq,T,d,d), k=matmul(h1,p.wk,T,d,d), v=matmul(h1,p.wv,T,d,d);
    // attention per head, causal
    var att=new Float32Array(h*T*T); var ctx=new Float32Array(T*d);
    var scale=1/Math.sqrt(hd);
    for(hh=0;hh<h;hh++){ var ho=hh*hd;
      for(t=0;t<T;t++){ var mx=-1e30; var base=hh*T*T+t*T;
        for(s=0;s<=t;s++){ var dot=0; for(i=0;i<hd;i++) dot+=q[t*d+ho+i]*k[s*d+ho+i]; dot*=scale; att[base+s]=dot; if(dot>mx)mx=dot; }
        var sum=0; for(s=0;s<=t;s++){ var e=Math.exp(att[base+s]-mx); att[base+s]=e; sum+=e; }
        for(s=0;s<=t;s++) att[base+s]/=sum;
        for(i=0;i<hd;i++){ var acc=0; for(s=0;s<=t;s++) acc+=att[base+s]*v[s*d+ho+i]; ctx[t*d+ho+i]=acc; }
      }
    }
    var attnout=matmul(ctx,p.wo,T,d,d);
    var x1=new Float32Array(T*d); for(i=0;i<T*d;i++) x1[i]=x0[i]+attnout[i];
    // ln2
    var ln2={mean:new Float32Array(T),inv:new Float32Array(T),xhat:new Float32Array(T*d)}, h2=new Float32Array(T*d);
    for(t=0;t<T;t++){ var m2=0; for(i=0;i<d;i++)m2+=x1[t*d+i]; m2/=d; var v2=0; for(i=0;i<d;i++){var dd=x1[t*d+i]-m2; v2+=dd*dd;} v2/=d; var inv2=1/Math.sqrt(v2+1e-5); ln2.mean[t]=m2; ln2.inv[t]=inv2; for(i=0;i<d;i++){ var xh2=(x1[t*d+i]-m2)*inv2; ln2.xhat[t*d+i]=xh2; h2[t*d+i]=xh2*p.ln2g[i]+p.ln2b[i]; } }
    // mlp
    var pre=new Float32Array(T*ff), f=new Float32Array(T*ff);
    for(t=0;t<T;t++)for(j=0;j<ff;j++){ var acc=p.bfc[j]; for(i=0;i<d;i++)acc+=h2[t*d+i]*p.wfc[j*d+i]; pre[t*ff+j]=acc; f[t*ff+j]=gelu(acc); }
    var mlp=new Float32Array(T*d);
    for(t=0;t<T;t++)for(i=0;i<d;i++){ var acc=p.bproj[i]; for(j=0;j<ff;j++)acc+=f[t*ff+j]*p.wproj[i*ff+j]; mlp[t*d+i]=acc; }
    var x2=new Float32Array(T*d); for(i=0;i<T*d;i++)x2[i]=x1[i]+mlp[i];
    // lnf
    var lnf={mean:new Float32Array(T),inv:new Float32Array(T),xhat:new Float32Array(T*d)}, hf=new Float32Array(T*d);
    for(t=0;t<T;t++){ var mf=0; for(i=0;i<d;i++)mf+=x2[t*d+i]; mf/=d; var vf=0; for(i=0;i<d;i++){var df=x2[t*d+i]-mf; vf+=df*df;} vf/=d; var invf=1/Math.sqrt(vf+1e-5); lnf.mean[t]=mf; lnf.inv[t]=invf; for(i=0;i<d;i++){ var xhf=(x2[t*d+i]-mf)*invf; lnf.xhat[t*d+i]=xhf; hf[t*d+i]=xhf*p.lnfg[i]+p.lnfb[i]; } }
    // logits (tied) + softmax CE
    var loss=0; var dlogits=null; if(wantGrad) dlogits=new Float32Array(T*V);
    var probsLast=null;
    for(t=0;t<T;t++){ var mxl=-1e30, vidx; var lg=new Float32Array(V);
      for(vidx=0;vidx<V;vidx++){ var acc=0; for(i=0;i<d;i++)acc+=hf[t*d+i]*p.wte[vidx*d+i]; lg[vidx]=acc; if(acc>mxl)mxl=acc; }
      var sum=0; for(vidx=0;vidx<V;vidx++){ var e=Math.exp(lg[vidx]-mxl); lg[vidx]=e; sum+=e; }
      for(vidx=0;vidx<V;vidx++) lg[vidx]/=sum;
      if(targets){ loss+=-Math.log(Math.max(1e-9,lg[targets[t]]));
        if(wantGrad) for(vidx=0;vidx<V;vidx++) dlogits[t*V+vidx]=(lg[vidx]-(vidx===targets[t]?1:0))/T; }
      if(t===T-1) probsLast=lg;
    }
    loss/=T;
    this.cache={ids:ids,x0:x0,ln1:ln1,h1:h1,q:q,k:k,v:v,att:att,ctx:ctx,attnout:attnout,x1:x1,ln2:ln2,h2:h2,pre:pre,f:f,x2:x2,lnf:lnf,hf:hf,dlogits:dlogits};
    return {loss:loss,probsLast:probsLast};
  };

  function matmul(A,W,T,din,dout){ // A [T,din], W [dout,din] row-major -> out [T,dout]
    var out=new Float32Array(T*dout),t,o,i;
    for(t=0;t<T;t++)for(o=0;o<dout;o++){ var acc=0; for(i=0;i<din;i++)acc+=A[t*din+i]*W[o*din+i]; out[t*dout+o]=acc; }
    return out;
  }

  MiniGPT.prototype.backwardGPT = function(){
    var d=this.d,h=this.h,hd=this.hd,V=this.V,ff=4*d,p=this.p,g=this.g,c=this.cache,T=c.ids.length,i,j,t,s,hh;
    var dhf=new Float32Array(T*d);
    // logits backward (tied wte)
    for(t=0;t<T;t++)for(var vidx=0;vidx<V;vidx++){ var dl=c.dlogits[t*V+vidx]; if(dl===0)continue; for(i=0;i<d;i++){ dhf[t*d+i]+=dl*p.wte[vidx*d+i]; g.wte[vidx*d+i]+=dl*c.hf[t*d+i]; } }
    // lnf backward
    var dx2=new Float32Array(T*d);
    lnBack(dhf,c.lnf,c.x2,p.lnfg,g.lnfg,g.lnfb,T,d,dx2);
    // residual x2 = x1 + mlp
    var dx1=new Float32Array(T*d), dmlp=new Float32Array(T*d);
    for(i=0;i<T*d;i++){ dx1[i]=dx2[i]; dmlp[i]=dx2[i]; }
    // mlp backward: mlp = f@wproj + bproj
    var df=new Float32Array(T*ff);
    for(t=0;t<T;t++)for(i=0;i<d;i++){ var dm=dmlp[t*d+i]; g.bproj[i]+=dm; for(j=0;j<ff;j++){ df[t*ff+j]+=dm*p.wproj[i*ff+j]; g.wproj[i*ff+j]+=dm*c.f[t*ff+j]; } }
    var dh2=new Float32Array(T*d);
    for(t=0;t<T;t++)for(j=0;j<ff;j++){ var dpre=df[t*ff+j]*dgelu(c.pre[t*ff+j]); g.bfc[j]+=dpre; for(i=0;i<d;i++){ dh2[t*d+i]+=dpre*p.wfc[j*d+i]; g.wfc[j*d+i]+=dpre*c.h2[t*d+i]; } }
    // ln2 backward -> dx1
    lnBack(dh2,c.ln2,c.x1,p.ln2g,g.ln2g,g.ln2b,T,d,dx1);
    // residual x1 = x0 + attnout
    var dx0=new Float32Array(T*d), dattnout=new Float32Array(T*d);
    for(i=0;i<T*d;i++){ dx0[i]=dx1[i]; dattnout[i]=dx1[i]; }
    // attnout = ctx @ wo
    var dctx=new Float32Array(T*d);
    for(t=0;t<T;t++)for(var o=0;o<d;o++){ var da=dattnout[t*d+o]; for(i=0;i<d;i++){ dctx[t*d+i]+=da*p.wo[o*d+i]; g.wo[o*d+i]+=da*c.ctx[t*d+i]; } }
    // attention backward
    var dq=new Float32Array(T*d), dk=new Float32Array(T*d), dv=new Float32Array(T*d);
    var scale=1/Math.sqrt(hd);
    for(hh=0;hh<h;hh++){ var ho=hh*hd;
      for(t=0;t<T;t++){ var base=hh*T*T+t*T;
        // dctx -> datt, dv
        var datt=new Float32Array(t+1);
        for(s=0;s<=t;s++){ var dot=0; for(i=0;i<hd;i++){ dot+=dctx[t*d+ho+i]*c.v[s*d+ho+i]; dv[s*d+ho+i]+=c.att[base+s]*dctx[t*d+ho+i]; } datt[s]=dot; }
        // softmax backward
        var dotsum=0; for(s=0;s<=t;s++) dotsum+=datt[s]*c.att[base+s];
        for(s=0;s<=t;s++){ var dscore=c.att[base+s]*(datt[s]-dotsum)*scale;
          for(i=0;i<hd;i++){ dq[t*d+ho+i]+=dscore*c.k[s*d+ho+i]; dk[s*d+ho+i]+=dscore*c.q[t*d+ho+i]; } }
      }
    }
    // q,k,v = h1 @ Wq/Wk/Wv  -> dh1, dW
    var dh1=new Float32Array(T*d);
    mmBack(dq,c.h1,p.wq,g.wq,dh1,T,d,d);
    mmBack(dk,c.h1,p.wk,g.wk,dh1,T,d,d);
    mmBack(dv,c.h1,p.wv,g.wv,dh1,T,d,d);
    // ln1 backward -> dx0
    lnBack(dh1,c.ln1,c.x0,p.ln1g,g.ln1g,g.ln1b,T,d,dx0);
    // embedding
    for(t=0;t<T;t++){ var id=c.ids[t]; for(i=0;i<d;i++){ g.wte[id*d+i]+=dx0[t*d+i]; g.wpe[t*d+i]+=dx0[t*d+i]; } }
  };

  function lnBack(dy, ln, x, g, gg, gb, T, d, dxOut){ // accumulates into dxOut
    for(var t=0;t<T;t++){ var off=t*d, inv=ln.inv[t];
      var dxhatDot=0, dyMean=0, i;
      for(i=0;i<d;i++){ var dxh=dy[off+i]*g[i]; gg[i]+=dy[off+i]*ln.xhat[off+i]; gb[i]+=dy[off+i]; dyMean+=dxh; dxhatDot+=dxh*ln.xhat[off+i]; }
      dyMean/=d; dxhatDot/=d;
      for(i=0;i<d;i++){ var dxh2=dy[off+i]*g[i]; dxOut[off+i]+=inv*(dxh2-dyMean-ln.xhat[off+i]*dxhatDot); }
    }
  }
  function mmBack(dOut, A, W, gW, dA, T, din, dout){ // out=A@W ; W[dout,din]; dOut[T,dout]
    for(var t=0;t<T;t++)for(var o=0;o<dout;o++){ var doo=dOut[t*dout+o]; if(doo===0)continue; for(var i=0;i<din;i++){ dA[t*din+i]+=doo*W[o*din+i]; gW[o*din+i]+=doo*A[t*din+i]; } }
  }

  MiniGPT.prototype.zeroGrad=function(){ var self=this; Object.keys(this.g).forEach(function(k){ self.g[k].fill(0); }); };
  MiniGPT.prototype.adam=function(scale){ var b1=0.9,b2=0.999,eps=1e-8,lr=this.lr,self=this; this.step++;
    var bc1=1-Math.pow(b1,this.step), bc2=1-Math.pow(b2,this.step);
    var P = this.arch==='rnn'? this.rp : this.p;
    Object.keys(P).forEach(function(key){ var p=P[key],g=self.g[key],m=self.m[key],v=self.v[key];
      for(var i=0;i<p.length;i++){ var gr=g[i]*scale; m[i]=b1*m[i]+(1-b1)*gr; v[i]=b2*v[i]+(1-b2)*gr*gr; p[i]-=lr*(m[i]/bc1)/(Math.sqrt(v[i]/bc2)+eps); } });
  };

  // single window: forward+backward, accumulates grad (no zero, no update)
  MiniGPT.prototype.trainWindow=function(){ var n=this.data.length,T=this.T; if(n<=T+1)return 0; var start=Math.floor(this.rand()*(n-T-1));
    var ids=this.data.subarray(start,start+T), tgt=this.data.subarray(start+1,start+T+1);
    if(this.arch==='rnn'){ var rr=this.forwardRNN(ids,tgt,true); this.backwardRNN(); return rr.loss; }
    var r=this.forwardGPT(ids,tgt,true); this.backwardGPT(); return r.loss; };

  // one optimizer step over `batch` random windows; returns avg loss
  MiniGPT.prototype.train=function(batch){
    batch=batch||8; this.zeroGrad(); var total=0;
    for(var b=0;b<batch;b++){ total+=this.trainWindow(); }
    this.adam(1/batch);
    this.tokensSeen=(this.tokensSeen||0)+batch*this.T;
    return total/batch;
  };

  MiniGPT.prototype.encodePrompt=function(prompt){ var ids=[],i; if(this.wordMode){ var ws=(prompt||'').toLowerCase().match(/[a-z']+|[.,;:!?-]/g)||[]; for(i=0;i<ws.length;i++){ var id=this.stoi[ws[i]]; if(id!==undefined)ids.push(id); } } else { for(i=0;i<(prompt||'').length;i++){ var c=this.stoi[prompt[i]]; if(c!==undefined)ids.push(c); } } return ids; };
  MiniGPT.prototype.decodeJoin=function(ids){ var s='',i; for(i=0;i<ids.length;i++){ s+=(i&&this.wordMode?' ':'')+this.chars[ids[i]]; } return s; };
  MiniGPT.prototype.pick=function(probs,temp){ var V=this.V,vidx,scaled=new Float32Array(V),sum=0; for(vidx=0;vidx<V;vidx++){ var pv=Math.pow(Math.max(1e-9,probs[vidx]),1/temp); scaled[vidx]=pv; sum+=pv; } var rr=this.rand()*sum,acc=0; for(vidx=0;vidx<V;vidx++){ acc+=scaled[vidx]; if(rr<=acc)return vidx; } return V-1; };

  MiniGPT.prototype.sample=function(nTok,temp,prompt){
    temp=temp||0.8; nTok=nTok||300; var T=this.T, d=this.d, i;
    var ctx=this.encodePrompt(prompt); if(!ctx.length)ctx.push(Math.floor(this.rand()*this.V));
    var outIds=ctx.slice();
    if(this.arch==='rnn'){
      var rp=this.rp, h=new Float32Array(d), j;
      var stepH=function(id){ var pre=new Float32Array(d); for(i=0;i<d;i++){ var acc=rp.r_bh[i]; for(j=0;j<d;j++)acc+=rp.r_wxh[i*d+j]*rp.r_emb[id*d+j]+rp.r_whh[i*d+j]*h[j]; pre[i]=acc; } for(i=0;i<d;i++)h[i]=Math.tanh(pre[i]); };
      for(i=0;i<ctx.length;i++)stepH(ctx[i]);
      var last=ctx[ctx.length-1];
      for(var s=0;s<nTok;s++){ var lg=new Float32Array(this.V),mx=-1e30,vidx; for(vidx=0;vidx<this.V;vidx++){ var a2=0; for(i=0;i<d;i++)a2+=h[i]*rp.r_emb[vidx*d+i]; lg[vidx]=a2; if(a2>mx)mx=a2; } var sum=0; for(vidx=0;vidx<this.V;vidx++){ var e=Math.exp(lg[vidx]-mx); lg[vidx]=e; sum+=e; } for(vidx=0;vidx<this.V;vidx++)lg[vidx]/=sum; var next=this.pick(lg,temp); outIds.push(next); stepH(next); last=next; }
      return this.decodeJoin(outIds);
    }
    for(var step=0;step<nTok;step++){
      var win=ctx.slice(Math.max(0,ctx.length-T));
      var r=this.forwardGPT(new Int32Array(win),null,false);
      var next=this.pick(r.probsLast,temp);
      ctx.push(next); outIds.push(next);
    }
    return this.decodeJoin(outIds);
  };

  MiniGPT.prototype.paramCount=function(){ var n=0,self=this,P=this.arch==='rnn'?this.rp:this.p; Object.keys(P).forEach(function(k){ n+=P[k].length; }); return n; };

  MiniGPT.prototype.forwardRNN=function(ids,targets,wantGrad){
    var d=this.d,T=ids.length,V=this.V,rp=this.rp,i,j,t,vidx;
    var H=new Float32Array(T*d), loss=0, dlogits=wantGrad?new Float32Array(T*V):null, probsLast=null;
    var hprev=new Float32Array(d);
    for(t=0;t<T;t++){ var id=ids[t];
      for(i=0;i<d;i++){ var acc=rp.r_bh[i]; for(j=0;j<d;j++)acc+=rp.r_wxh[i*d+j]*rp.r_emb[id*d+j]+rp.r_whh[i*d+j]*hprev[j]; H[t*d+i]=Math.tanh(acc); }
      var lg=new Float32Array(V),mx=-1e30; for(vidx=0;vidx<V;vidx++){ var a2=0; for(i=0;i<d;i++)a2+=H[t*d+i]*rp.r_emb[vidx*d+i]; lg[vidx]=a2; if(a2>mx)mx=a2; }
      var sum=0; for(vidx=0;vidx<V;vidx++){ var e=Math.exp(lg[vidx]-mx); lg[vidx]=e; sum+=e; } for(vidx=0;vidx<V;vidx++)lg[vidx]/=sum;
      if(targets){ loss+=-Math.log(Math.max(1e-9,lg[targets[t]])); if(wantGrad)for(vidx=0;vidx<V;vidx++)dlogits[t*V+vidx]=(lg[vidx]-(vidx===targets[t]?1:0))/T; }
      if(t===T-1)probsLast=lg; hprev=H.subarray(t*d,t*d+d);
    }
    loss/=T; this.rcache={ids:ids,H:H,dlogits:dlogits}; return {loss:loss,probsLast:probsLast};
  };
  MiniGPT.prototype.backwardRNN=function(){
    var d=this.d,V=this.V,rp=this.rp,g=this.g,c=this.rcache,T=c.ids.length,i,j,t,vidx;
    var dhnext=new Float32Array(d);
    for(t=T-1;t>=0;t--){ var id=c.ids[t]; var dh=new Float32Array(d);
      for(vidx=0;vidx<V;vidx++){ var dl=c.dlogits[t*V+vidx]; if(dl===0)continue; for(i=0;i<d;i++){ dh[i]+=dl*rp.r_emb[vidx*d+i]; g.r_emb[vidx*d+i]+=dl*c.H[t*d+i]; } }
      for(i=0;i<d;i++)dh[i]+=dhnext[i];
      var dpre=new Float32Array(d); for(i=0;i<d;i++){ var hv=c.H[t*d+i]; dpre[i]=dh[i]*(1-hv*hv); g.r_bh[i]+=dpre[i]; }
      var hprev = t>0? c.H.subarray((t-1)*d,(t-1)*d+d) : null;
      for(i=0;i<d;i++){ var dp=dpre[i]; for(j=0;j<d;j++){ g.r_wxh[i*d+j]+=dp*rp.r_emb[id*d+j]; g.r_emb[id*d+j]+=dp*rp.r_wxh[i*d+j]; g.r_whh[i*d+j]+=dp*(hprev?hprev[j]:0); } }
      var dhp=new Float32Array(d); for(j=0;j<d;j++){ var acc=0; for(i=0;i<d;i++)acc+=rp.r_whh[i*d+j]*dpre[i]; dhp[j]=acc; } dhnext=dhp;
    }
  };

  // ---- embeddings (token vectors) ----
  MiniGPT.prototype.embMatrix=function(){ var vecs=this.arch==='rnn'?this.rp.r_emb:this.p.wte; return {toks:this.chars, V:this.V, d:this.d, vecs:vecs}; };
  MiniGPT.prototype.cos=function(a,ao,b,bo,d){ var dot=0,na=0,nb=0,i; for(i=0;i<d;i++){ dot+=a[ao+i]*b[bo+i]; na+=a[ao+i]*a[ao+i]; nb+=b[bo+i]*b[bo+i]; } return dot/(Math.sqrt(na*nb)+1e-9); };
  MiniGPT.prototype.nearestVec=function(vec,exclude,n){ var e=this.embMatrix(),d=e.d,V=e.V,res=[],i; exclude=exclude||{}; 
    var nv=0; for(i=0;i<d;i++)nv+=vec[i]*vec[i]; nv=Math.sqrt(nv)+1e-9;
    for(i=0;i<V;i++){ if(exclude[i])continue; var dot=0,nb=0,j; for(j=0;j<d;j++){ dot+=vec[j]*e.vecs[i*d+j]; nb+=e.vecs[i*d+j]*e.vecs[i*d+j]; } res.push({i:i,tok:e.toks[i],sim:dot/(nv*(Math.sqrt(nb)+1e-9))}); }
    res.sort(function(a,b){return b.sim-a.sim;}); return res.slice(0,n||6); };
  MiniGPT.prototype.analogy=function(aTok,bTok,cTok,n){ var e=this.embMatrix(),d=e.d; var a=this.stoi[aTok],b=this.stoi[bTok],c=this.stoi[cTok]; if(a===undefined||b===undefined||c===undefined)return null;
    var vec=new Float32Array(d),i; for(i=0;i<d;i++)vec[i]=e.vecs[a*d+i]-e.vecs[b*d+i]+e.vecs[c*d+i];
    var ex={}; ex[a]=1;ex[b]=1;ex[c]=1; return this.nearestVec(vec,ex,n||5); };
  MiniGPT.prototype.pca=function(k){ var e=this.embMatrix(),d=e.d,V=e.V,i,j,t; k=k||2;
    var mean=new Float32Array(d); for(i=0;i<V;i++)for(j=0;j<d;j++)mean[j]+=e.vecs[i*d+j]; for(j=0;j<d;j++)mean[j]/=V;
    var X=new Float32Array(V*d); for(i=0;i<V;i++)for(j=0;j<d;j++)X[i*d+j]=e.vecs[i*d+j]-mean[j];
    var C=new Float32Array(d*d); for(i=0;i<V;i++)for(j=0;j<d;j++){ var xj=X[i*d+j]; for(var l=0;l<d;l++)C[j*d+l]+=xj*X[i*d+l]; }
    for(i=0;i<d*d;i++)C[i]/=V;
    var comps=[]; var Cw=C.slice();
    for(var c=0;c<k;c++){ var vvec=new Float32Array(d); for(j=0;j<d;j++)vvec[j]=Math.random()-0.5;
      for(t=0;t<80;t++){ var nw=new Float32Array(d); for(j=0;j<d;j++){ var acc=0; for(l=0;l<d;l++)acc+=Cw[j*d+l]*vvec[l]; nw[j]=acc; } var nrm=0; for(j=0;j<d;j++)nrm+=nw[j]*nw[j]; nrm=Math.sqrt(nrm)+1e-9; for(j=0;j<d;j++)vvec[j]=nw[j]/nrm; }
      // eigenvalue
      var Cv=new Float32Array(d); for(j=0;j<d;j++){ var acc=0; for(l=0;l<d;l++)acc+=C[j*d+l]*vvec[l]; Cv[j]=acc; } var lam=0; for(j=0;j<d;j++)lam+=vvec[j]*Cv[j];
      comps.push(vvec);
      for(j=0;j<d;j++)for(l=0;l<d;l++)Cw[j*d+l]-=lam*vvec[j]*vvec[l]; // deflate
    }
    var coords=new Float32Array(V*k); for(i=0;i<V;i++)for(c=0;c<k;c++){ var acc=0; for(j=0;j<d;j++)acc+=X[i*d+j]*comps[c][j]; coords[i*k+c]=acc; }
    return {coords:coords, k:k, toks:e.toks, V:V};
  };

  global.MiniGPT = MiniGPT;
})(window);
