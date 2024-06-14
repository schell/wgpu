//! [`Module`] helpers for "upgrading" atomics in the SPIR-V (and eventually GLSL) frontends.
use std::{
    collections::VecDeque,
    sync::{atomic::AtomicUsize, Arc},
};

use rustc_hash::FxHashMap;

use crate::{
    Constant, Expression, Function, FunctionArgument, GlobalVariable, Handle, LocalVariable,
    Module, Override, StructMember, Type, TypeInner,
};

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("bad handle: {0}")]
    MissingHandle(crate::arena::BadHandle),
    #[error("no function context")]
    NoFunction,
    #[error("function {0:?} is missing arg {1}")]
    MissingFnArg(Handle<Function>, usize),
    #[error("encountered an unsupported expression")]
    Unsupported,
}

impl From<Error> for crate::front::spv::Error {
    fn from(source: Error) -> Self {
        crate::front::spv::Error::AtomicUpgradeError(source)
    }
}

impl From<crate::arena::BadHandle> for Error {
    fn from(value: crate::arena::BadHandle) -> Self {
        Error::MissingHandle(value)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum AtomicOpInst {
    AtomicIIncrement,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AtomicOp {
    pub instruction: AtomicOpInst,
    /// Handle to the pointer's type in the module
    pub pointer_type_handle: Handle<Type>,
    /// Handle to the pointer expression in the module/function
    pub pointer_handle: Handle<Expression>,
}

#[derive(Clone, Default)]
struct Padding(Arc<AtomicUsize>);

impl std::fmt::Display for Padding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for _ in 0..self.0.load(std::sync::atomic::Ordering::Relaxed) {
            f.write_str("  ")?;
        }
        Ok(())
    }
}

impl Drop for Padding {
    fn drop(&mut self) {
        let _ = self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Padding {
    fn trace(&self, msg: impl std::fmt::Display, t: impl std::fmt::Debug) {
        format!("{msg} {t:#?}")
            .split('\n')
            .for_each(|ln| log::trace!("{self}{ln}"));
    }

    fn debug(&self, msg: impl std::fmt::Display, t: impl std::fmt::Debug) {
        format!("{msg} {t:#?}")
            .split('\n')
            .for_each(|ln| log::debug!("{self}{ln}"));
    }

    fn inc_padding(&self) -> Padding {
        let _ = self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum AnyHandle {
    Type(Handle<Type>),
    GblExpr(Handle<Expression>),
    GblVar(Handle<GlobalVariable>),
    Constant(Handle<Constant>),
    Override(Handle<Override>),
    FnExpr(Handle<Function>, Handle<Expression>),
    FnVar(Handle<Function>, Handle<LocalVariable>),
    FnArg(Handle<Function>, usize),
}

impl From<Handle<Type>> for AnyHandle {
    fn from(value: Handle<Type>) -> Self {
        Self::Type(value)
    }
}

impl From<Handle<Expression>> for AnyHandle {
    fn from(value: Handle<Expression>) -> Self {
        Self::GblExpr(value)
    }
}

impl From<Handle<GlobalVariable>> for AnyHandle {
    fn from(value: Handle<GlobalVariable>) -> Self {
        Self::GblVar(value)
    }
}

impl From<Handle<Constant>> for AnyHandle {
    fn from(value: Handle<Constant>) -> Self {
        Self::Constant(value)
    }
}

impl From<Handle<Override>> for AnyHandle {
    fn from(value: Handle<Override>) -> Self {
        Self::Override(value)
    }
}

impl From<(Option<Handle<Function>>, Handle<Expression>)> for AnyHandle {
    fn from((mfh, h): (Option<Handle<Function>>, Handle<Expression>)) -> Self {
        if let Some(fh) = mfh {
            Self::FnExpr(fh, h)
        } else {
            Self::GblExpr(h)
        }
    }
}

impl From<(Handle<Function>, Handle<Expression>)> for AnyHandle {
    fn from((fh, h): (Handle<Function>, Handle<Expression>)) -> Self {
        Self::FnExpr(fh, h)
    }
}

impl From<(Handle<Function>, Handle<LocalVariable>)> for AnyHandle {
    fn from((fh, h): (Handle<Function>, Handle<LocalVariable>)) -> Self {
        Self::FnVar(fh, h)
    }
}

impl From<(Handle<Function>, usize)> for AnyHandle {
    fn from((fh, i): (Handle<Function>, usize)) -> Self {
        Self::FnArg(fh, i)
    }
}

impl AnyHandle {
    const fn to_type(self) -> Option<Handle<Type>> {
        match self {
            AnyHandle::Type(h) => Some(h),
            _ => None,
        }
    }

    const fn to_override(self) -> Option<Handle<Override>> {
        match self {
            AnyHandle::Override(h) => Some(h),
            _ => None,
        }
    }

    const fn to_constant(self) -> Option<Handle<Constant>> {
        match self {
            AnyHandle::Constant(h) => Some(h),
            _ => None,
        }
    }

    const fn to_gbl_var(self) -> Option<Handle<GlobalVariable>> {
        match self {
            AnyHandle::GblVar(h) => Some(h),
            _ => None,
        }
    }

    const fn to_expr(self) -> Option<Handle<Expression>> {
        match self {
            AnyHandle::GblExpr(h) => Some(h),
            AnyHandle::FnExpr(_, h) => Some(h),
            _ => None,
        }
    }

    const fn to_fn_var(self) -> Option<Handle<LocalVariable>> {
        match self {
            AnyHandle::FnVar(_fh, h) => Some(h),
            _ => None,
        }
    }
}

struct UpgradeState<'a> {
    padding: Padding,
    module: &'a mut Module,
    upgraded_handles: FxHashMap<AnyHandle, AnyHandle>,
    unresolved_handles: VecDeque<(AnyHandle, AnyHandle)>,
    needle_and_replacement: Option<(AnyHandle, AnyHandle)>,
}

impl<'a> UpgradeState<'a> {
    /// Increment the log padding level.
    fn inc_padding(&self) -> Padding {
        self.padding.inc_padding()
    }

    /// Add the new symbol as an upgrade from a previous one.
    fn add_upgraded(
        &mut self,
        prev_any_handle: impl Into<AnyHandle>,
        new_any_handle: impl Into<AnyHandle>,
    ) {
        let prev_any_handle = prev_any_handle.into();
        let new_any_handle = new_any_handle.into();
        log::debug!(
            "{}adding upgrade: {prev_any_handle:?} => {new_any_handle:?}",
            self.padding
        );
        self.upgraded_handles
            .insert(prev_any_handle, new_any_handle);
        self.unresolved_handles
            .push_front((prev_any_handle, new_any_handle));
    }

    /// Instead of always recreating new symbols, we can return a previously upgraded
    /// symbol. If we do not, the entire process gets stuck in an infinite loop of
    /// upgrades.
    ///
    /// > But what about when we want to resolve newly added upgrades within a previously
    /// upgraded symbol?
    ///
    /// For those cases I think it's fine to only recurse into that
    /// newly created symbol and not the symbol it was upgraded _from_. In this way
    /// each newly created symbol will be one step/change from the one it was upgraded
    /// from, and eventually all things that need replacing will be replaced - albeit
    /// with many, many extra symbols strewn about the `Module`, which is fine.
    fn get_upgraded(&self, haystack: impl Into<AnyHandle>) -> Option<AnyHandle> {
        let haystack = haystack.into();
        let replacement = self.upgraded_handles.get(&haystack)?;
        log::trace!(
            "{}{haystack:?} previously upgraded to {replacement:?}",
            self.padding
        );
        Some(*replacement)
    }

    /// Dequeue the the next unresolved upgrade handle.
    fn pop_unresolved(&mut self) -> Option<(AnyHandle, AnyHandle)> {
        self.unresolved_handles.pop_back()
    }

    /// Compare `haystack` to the current search "needle" and if they are equal, return
    /// the replacement.
    fn compare_to_search(&self, haystack: impl Into<AnyHandle>) -> Option<AnyHandle> {
        let (needle, replacement) = self.needle_and_replacement?;
        if needle == haystack.into() {
            log::trace!("replacing {needle:?} => {replacement:?}");
            Some(replacement)
        } else {
            None
        }
    }

    /// Upgrade the type, recursing until we reach the leaves.
    /// At the leaves, replace scalars with atomic scalars.
    ///
    /// NOTE: we *don't* automatically add types to the list of upgrades because
    /// not all types need to be resolved and replaced everywhere.
    /// For example, consider the implications of doing this when encountering the
    /// scalar `u32` type: every reference of `u32` would be replaced with an atomic,
    /// which is obviously incorrect.   
    fn upgrade_type(&mut self, type_handle: Handle<Type>) -> Result<Handle<Type>, Error> {
        let padding = self.inc_padding();

        if let Some(replacement) = self
            .compare_to_search(type_handle)
            .and_then(|h| h.to_type())
        {
            return Ok(replacement);
        }

        padding.trace("upgrading type: ", type_handle);
        let type_ = self
            .module
            .types
            .get_handle(type_handle)
            .map_err(Error::MissingHandle)?
            .clone();

        let new_inner = match type_.inner.clone() {
            TypeInner::Scalar(scalar) => TypeInner::Atomic(scalar),
            TypeInner::Pointer { base, space } => TypeInner::Pointer {
                base: self.upgrade_type(base)?,
                space,
            },
            TypeInner::Array { base, size, stride } => TypeInner::Array {
                base: self.upgrade_type(base)?,
                size,
                stride,
            },
            TypeInner::Struct { members, span } => TypeInner::Struct {
                members: {
                    let mut new_members = vec![];
                    for member in members.iter().cloned() {
                        let StructMember {
                            name,
                            ty,
                            binding,
                            offset,
                        } = member;
                        new_members.push(StructMember {
                            name,
                            ty: self.upgrade_type(ty)?,
                            binding,
                            offset,
                        });
                    }
                    new_members
                },
                span,
            },
            TypeInner::BindingArray { base, size } => TypeInner::BindingArray {
                base: self.upgrade_type(base)?,
                size,
            },
            n => n,
        };

        let new_type = Type {
            name: type_.name.clone(),
            inner: new_inner,
        };
        let new_handle = if let Some(handle) = self.module.types.get(&new_type) {
            padding.trace("type exists: ", handle);
            handle
        } else {
            padding.debug("type: ", type_handle);
            padding.debug("from: ", &type_);
            padding.debug("to:   ", &new_type);

            let new_handle = self
                .module
                .types
                .insert(new_type, self.module.types.get_span(type_handle));
            padding.debug("h: ", new_handle);
            new_handle
        };

        Ok(new_handle)
    }

    fn upgrade_global_variable(
        &mut self,
        handle: Handle<GlobalVariable>,
    ) -> Result<Handle<GlobalVariable>, Error> {
        let padding = self.inc_padding();

        if let Some(replacement) = self.compare_to_search(handle).and_then(|h| h.to_gbl_var()) {
            return Ok(replacement);
        }

        if let Some(replacement) = self.get_upgraded(handle).and_then(|h| h.to_gbl_var()) {
            return Ok(replacement);
        }

        padding.trace("upgrading global variable: ", handle);

        let var = self.module.global_variables.try_get(handle)?.clone();
        let new_var = GlobalVariable {
            name: var.name.clone(),
            space: var.space,
            binding: var.binding.clone(),
            ty: self.upgrade_type(var.ty)?,
            init: self.upgrade_opt_expression(None, var.init)?,
        };
        if new_var != var {
            padding.debug("global var:     ", &var);
            padding.debug("new global var: ", &new_var);
            let span = self.module.global_variables.get_span(handle);
            let new_handle = self.module.global_variables.append(new_var, span);
            self.add_upgraded(handle, new_handle);
            Ok(new_handle)
        } else {
            Ok(handle)
        }
    }

    fn upgrade_local_variable(
        &mut self,
        fn_handle: Handle<Function>,
        handle: Handle<LocalVariable>,
    ) -> Result<Handle<LocalVariable>, Error> {
        let padding = self.inc_padding();

        if let Some(replacement) = self
            .compare_to_search((fn_handle, handle))
            .and_then(|h| h.to_fn_var())
        {
            return Ok(replacement);
        }

        if let Some(replacement) = self
            .get_upgraded((fn_handle, handle))
            .and_then(|h| h.to_fn_var())
        {
            return Ok(replacement);
        }

        padding.trace("upgrading local variable: ", handle);

        let (var, span) = {
            let f = self.module.functions.try_get(fn_handle)?;
            let var = f.local_variables.try_get(handle)?.clone();
            let span = f.local_variables.get_span(handle);
            (var, span)
        };

        let new_var = LocalVariable {
            name: var.name.clone(),
            ty: self.upgrade_type(var.ty)?,
            init: self.upgrade_opt_expression(Some(fn_handle), var.init)?,
        };
        if new_var != var {
            padding.debug("local var:     ", &var);
            padding.debug("new local var: ", &new_var);
            let f = self.module.functions.get_mut(fn_handle);
            let new_handle = f.local_variables.append(new_var, span);
            self.add_upgraded((fn_handle, handle), (fn_handle, new_handle));
            Ok(new_handle)
        } else {
            Ok(handle)
        }
    }

    fn upgrade_constant(&mut self, handle: Handle<Constant>) -> Result<Handle<Constant>, Error> {
        let padding = self.inc_padding();

        if let Some(replacement) = self.compare_to_search(handle).and_then(|h| h.to_constant()) {
            return Ok(replacement);
        }

        if let Some(replacement) = self.get_upgraded(handle).and_then(|h| h.to_constant()) {
            return Ok(replacement);
        }

        padding.trace("upgrading const: ", handle);

        let constant = self.module.constants.try_get(handle)?.clone();

        let new_constant = Constant {
            name: constant.name.clone(),
            ty: self.upgrade_type(constant.ty)?,
            init: self.upgrade_expression(None, constant.init)?,
        };

        if constant != new_constant {
            padding.debug("constant:     ", &constant);
            padding.debug("new constant: ", &new_constant);
            let new_handle = self
                .module
                .constants
                .append(new_constant, self.module.constants.get_span(handle));
            self.add_upgraded(handle, new_handle);
            Ok(new_handle)
        } else {
            Ok(handle)
        }
    }

    fn upgrade_override(&mut self, handle: Handle<Override>) -> Result<Handle<Override>, Error> {
        let padding = self.inc_padding();

        if let Some(replacement) = self.compare_to_search(handle).and_then(|h| h.to_override()) {
            return Ok(replacement);
        }

        if let Some(replacement) = self.get_upgraded(handle).and_then(|h| h.to_override()) {
            return Ok(replacement);
        }

        padding.trace("upgrading override: ", handle);

        let o = self.module.overrides.try_get(handle)?.clone();

        let new_o = Override {
            name: o.name.clone(),
            id: o.id,
            ty: self.upgrade_type(o.ty)?,
            init: self.upgrade_opt_expression(None, o.init)?,
        };

        if o != new_o {
            padding.debug("override:     ", &o);
            padding.debug("new override: ", &new_o);
            let new_handle = self
                .module
                .overrides
                .append(new_o, self.module.overrides.get_span(handle));
            self.add_upgraded(handle, new_handle);
            Ok(new_handle)
        } else {
            Ok(handle)
        }
    }

    fn upgrade_opt_expression(
        &mut self,
        maybe_fn_handle: Option<Handle<Function>>,
        maybe_handle: Option<Handle<Expression>>,
    ) -> Result<Option<Handle<Expression>>, Error> {
        Ok(if let Some(h) = maybe_handle {
            Some(self.upgrade_expression(maybe_fn_handle, h)?)
        } else {
            None
        })
    }

    fn upgrade_expression(
        &mut self,
        maybe_fn_handle: Option<Handle<Function>>,
        handle: Handle<Expression>,
    ) -> Result<Handle<Expression>, Error> {
        let padding = self.inc_padding();

        if let Some(replacement) = self
            .compare_to_search((maybe_fn_handle, handle))
            .and_then(|h| h.to_expr())
        {
            return Ok(replacement);
        }

        if let Some(replacement) = self
            .get_upgraded((maybe_fn_handle, handle))
            .and_then(|h| h.to_expr())
        {
            return Ok(replacement);
        }

        padding.trace("upgrading expr: ", handle);
        let expr = if let Some(fh) = maybe_fn_handle {
            let function = self.module.functions.try_get(fh)?;
            function.expressions.try_get(handle)?.clone()
        } else {
            self.module.global_expressions.try_get(handle)?.clone()
        };

        let new_expr = match expr.clone() {
            l @ Expression::Literal(_) => l,
            Expression::Constant(h) => Expression::Constant(self.upgrade_constant(h)?),
            Expression::Override(h) => Expression::Override(self.upgrade_override(h)?),
            Expression::ZeroValue(ty) => Expression::ZeroValue(self.upgrade_type(ty)?),
            Expression::Compose { ty, components } => Expression::Compose {
                ty: self.upgrade_type(ty)?,
                components: {
                    let mut new_components = vec![];
                    for component in components.into_iter() {
                        new_components.push(self.upgrade_expression(maybe_fn_handle, component)?);
                    }
                    new_components
                },
            },
            Expression::Access { base, index } => Expression::Access {
                base: self.upgrade_expression(maybe_fn_handle, base)?,
                index: self.upgrade_expression(maybe_fn_handle, index)?,
            },
            Expression::AccessIndex { base, index } => Expression::AccessIndex {
                base: self.upgrade_expression(maybe_fn_handle, base)?,
                index,
            },
            Expression::Splat { size, value } => Expression::Splat {
                size,
                value: self.upgrade_expression(maybe_fn_handle, value)?,
            },
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => Expression::Swizzle {
                size,
                vector: self.upgrade_expression(maybe_fn_handle, vector)?,
                pattern,
            },
            Expression::FunctionArgument(index) => {
                let fn_handle = maybe_fn_handle.ok_or(Error::NoFunction)?;
                self.upgrade_fn_arg(fn_handle, index as usize)?;
                Expression::FunctionArgument(index)
            }
            Expression::GlobalVariable(var) => {
                Expression::GlobalVariable(self.upgrade_global_variable(var)?)
            }
            Expression::LocalVariable(var) => Expression::LocalVariable(
                self.upgrade_local_variable(maybe_fn_handle.ok_or(Error::NoFunction)?, var)?,
            ),
            Expression::Load { pointer } => Expression::Load {
                pointer: self.upgrade_expression(maybe_fn_handle, pointer)?,
            },
            Expression::ImageSample {
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => Expression::ImageSample {
                image: self.upgrade_expression(maybe_fn_handle, image)?,
                sampler: self.upgrade_expression(maybe_fn_handle, sampler)?,
                gather,
                coordinate: self.upgrade_expression(maybe_fn_handle, coordinate)?,
                array_index: self.upgrade_opt_expression(maybe_fn_handle, array_index)?,
                offset: self.upgrade_opt_expression(maybe_fn_handle, offset)?,
                level: match level {
                    crate::SampleLevel::Exact(h) => {
                        crate::SampleLevel::Exact(self.upgrade_expression(maybe_fn_handle, h)?)
                    }
                    crate::SampleLevel::Bias(h) => {
                        crate::SampleLevel::Bias(self.upgrade_expression(maybe_fn_handle, h)?)
                    }
                    crate::SampleLevel::Gradient { x, y } => crate::SampleLevel::Gradient {
                        x: self.upgrade_expression(maybe_fn_handle, x)?,
                        y: self.upgrade_expression(maybe_fn_handle, y)?,
                    },
                    n => n,
                },
                depth_ref: self.upgrade_opt_expression(maybe_fn_handle, depth_ref)?,
            },
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => Expression::ImageLoad {
                image: self.upgrade_expression(maybe_fn_handle, image)?,
                coordinate: self.upgrade_expression(maybe_fn_handle, coordinate)?,
                array_index: self.upgrade_opt_expression(maybe_fn_handle, array_index)?,
                sample: self.upgrade_opt_expression(maybe_fn_handle, sample)?,
                level: self.upgrade_opt_expression(maybe_fn_handle, level)?,
            },
            Expression::ImageQuery { image, query } => Expression::ImageQuery {
                image: self.upgrade_expression(maybe_fn_handle, image)?,
                query: match query {
                    crate::ImageQuery::Size { level } => crate::ImageQuery::Size {
                        level: self.upgrade_opt_expression(maybe_fn_handle, level)?,
                    },
                    n => n,
                },
            },
            Expression::Unary { op, expr } => Expression::Unary {
                op,
                expr: self.upgrade_expression(maybe_fn_handle, expr)?,
            },
            Expression::Binary { op, left, right } => Expression::Binary {
                op,
                left: self.upgrade_expression(maybe_fn_handle, left)?,
                right: self.upgrade_expression(maybe_fn_handle, right)?,
            },
            Expression::Select {
                condition,
                accept,
                reject,
            } => Expression::Select {
                condition: self.upgrade_expression(maybe_fn_handle, condition)?,
                accept: self.upgrade_expression(maybe_fn_handle, accept)?,
                reject: self.upgrade_expression(maybe_fn_handle, reject)?,
            },
            Expression::Derivative { axis, ctrl, expr } => Expression::Derivative {
                axis,
                ctrl,
                expr: self.upgrade_expression(maybe_fn_handle, expr)?,
            },
            Expression::Relational { fun, argument } => Expression::Relational {
                fun,
                argument: self.upgrade_expression(maybe_fn_handle, argument)?,
            },
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => Expression::Math {
                fun,
                arg: self.upgrade_expression(maybe_fn_handle, arg)?,
                arg1: self.upgrade_opt_expression(maybe_fn_handle, arg1)?,
                arg2: self.upgrade_opt_expression(maybe_fn_handle, arg2)?,
                arg3: self.upgrade_opt_expression(maybe_fn_handle, arg3)?,
            },
            Expression::As {
                expr,
                kind,
                convert,
            } => Expression::As {
                expr: self.upgrade_expression(maybe_fn_handle, expr)?,
                kind,
                convert,
            },
            c @ Expression::CallResult(_) => c,
            a @ Expression::AtomicResult { .. } => a,
            Expression::WorkGroupUniformLoadResult { ty } => {
                Expression::WorkGroupUniformLoadResult {
                    ty: self.upgrade_type(ty)?,
                }
            }
            Expression::ArrayLength(h) => {
                Expression::ArrayLength(self.upgrade_expression(maybe_fn_handle, h)?)
            }
            r @ Expression::RayQueryProceedResult => r,
            Expression::RayQueryGetIntersection { query, committed } => {
                Expression::RayQueryGetIntersection {
                    query: self.upgrade_expression(maybe_fn_handle, query)?,
                    committed,
                }
            }
            s @ Expression::SubgroupBallotResult => s,
            Expression::SubgroupOperationResult { ty } => Expression::SubgroupOperationResult {
                ty: self.upgrade_type(ty)?,
            },
        };

        if new_expr != expr {
            padding.debug("expr    : ", &expr);
            padding.trace("new expr: ", &new_expr);
            let arena = if let Some(fh) = maybe_fn_handle {
                let f = self.module.functions.get_mut(fh);
                &mut f.expressions
            } else {
                &mut self.module.global_expressions
            };
            let span = arena.get_span(handle);
            let new_handle = arena.append(new_expr, span);
            self.add_upgraded((maybe_fn_handle, handle), (maybe_fn_handle, new_handle));
            Ok(new_handle)
        } else {
            Ok(handle)
        }
    }

    /// Upgrade the function argument, which results in an in-place modification of the argument.
    fn upgrade_fn_arg(&mut self, fn_handle: Handle<Function>, index: usize) -> Result<(), Error> {
        let padding = self.inc_padding();
        padding.trace("upgrading: ", (fn_handle, index));

        let arg = {
            let f = self.module.functions.try_get(fn_handle)?;
            f.arguments
                .get(index)
                .ok_or(Error::MissingFnArg(fn_handle, index))?
                .clone()
        };

        let new_arg = FunctionArgument {
            name: arg.name.clone(),
            // TODO: we possibly only want to do the type upgrade if we're searching for this type
            ty: self.upgrade_type(arg.ty)?,
            binding: arg.binding.clone(),
        };

        if new_arg != arg {
            padding.debug("fn arg:     ", &arg);
            padding.trace("new fn arg: ", &new_arg);
            let f = self.module.functions.get_mut(fn_handle);
            *f.arguments
                .get_mut(index)
                .ok_or(Error::MissingFnArg(fn_handle, index))? = new_arg;
            self.add_upgraded((fn_handle, index), (fn_handle, index));
        }

        Ok(())
    }

    /// Returns all the possible handles that might contain `needle`.
    fn get_haystacks(&self, needle: AnyHandle) -> Vec<AnyHandle> {
        fn add_fn_exprs(module: &Module, handles: &mut Vec<AnyHandle>, fh: Handle<Function>) {
            if let Ok(f) = module.functions.try_get(fh) {
                handles.extend(
                    f.expressions
                        .iter_handles()
                        .map(|h| AnyHandle::from((fh, h))),
                );
            }
        }

        fn add_fn_vars(module: &Module, handles: &mut Vec<AnyHandle>, fh: Handle<Function>) {
            if let Ok(f) = module.functions.try_get(fh) {
                handles.extend(
                    f.local_variables
                        .iter_handles()
                        .map(|h| AnyHandle::from((fh, h))),
                );
            }
        }

        fn add_fn_args(module: &Module, handles: &mut Vec<AnyHandle>, fh: Handle<Function>) {
            if let Ok(f) = module.functions.try_get(fh) {
                handles.extend((0..f.arguments.len()).map(|i| AnyHandle::from((fh, i))));
            }
        }

        let mut handles: Vec<AnyHandle> = vec![];
        match needle {
            // Anything could have a reference to a global expression
            // ...and expressions may contain global vars,
            //    therefore anything may contain global vars
            // ...and expressions may contain constants,
            //    therefore anything may contain constants
            // ...expressions may contain overrides,
            //    therefore anything may contain overrides
            //
            // Also, types are treated as a "global" thing and are found in all
            // other things, but we don't need to search types, as their handles
            // are included in the AST as we encounter them.
            AnyHandle::GblExpr(_)
            | AnyHandle::GblVar(_)
            | AnyHandle::Constant(_)
            | AnyHandle::Override(_)
            | AnyHandle::Type(_) => {
                handles.extend(
                    self.module
                        .global_expressions
                        .iter_handles()
                        .map(AnyHandle::from),
                );
                handles.extend(
                    self.module
                        .global_variables
                        .iter_handles()
                        .map(AnyHandle::from),
                );
                handles.extend(self.module.constants.iter_handles().map(AnyHandle::from));
                handles.extend(self.module.overrides.iter_handles().map(AnyHandle::from));
                for fh in self.module.functions.iter_handles() {
                    add_fn_exprs(self.module, &mut handles, fh);
                }
                for fh in self.module.functions.iter_handles() {
                    add_fn_vars(self.module, &mut handles, fh);
                }
                for fh in self.module.functions.iter_handles() {
                    add_fn_args(self.module, &mut handles, fh);
                }
            }
            // Whereas function expressions, vars and args will *not* be found inside global
            // things, and furthermore will *only* be found within one function
            AnyHandle::FnExpr(fh, _) | AnyHandle::FnVar(fh, _) | AnyHandle::FnArg(fh, _) => {
                add_fn_exprs(self.module, &mut handles, fh);
                add_fn_vars(self.module, &mut handles, fh);
            }
        }

        handles
    }

    fn search(&mut self, haystack: AnyHandle) -> Result<(), Error> {
        match haystack {
            AnyHandle::Type(_) => {
                // We never use type handles as haystacks, you can verify this by looking
                // at Upgrade::get_haystacks
                unreachable!("types aren't used as haystacks");
            }
            AnyHandle::GblExpr(h) => {
                let _ = self.upgrade_expression(None, h)?;
            }
            AnyHandle::GblVar(h) => {
                let _ = self.upgrade_global_variable(h)?;
            }
            AnyHandle::Constant(h) => {
                let _ = self.upgrade_constant(h)?;
            }
            AnyHandle::Override(h) => {
                let _ = self.upgrade_override(h)?;
            }
            AnyHandle::FnExpr(fh, h) => {
                let _ = self.upgrade_expression(Some(fh), h)?;
            }
            AnyHandle::FnVar(fh, h) => {
                let _ = self.upgrade_local_variable(fh, h)?;
            }
            AnyHandle::FnArg(fh, i) => self.upgrade_fn_arg(fh, i)?,
        };
        Ok(())
    }
}

impl Module {
    /// Upgrade all atomics given.
    pub(crate) fn upgrade_atomics(
        &mut self,
        ops: impl IntoIterator<Item = AtomicOp>,
    ) -> Result<(), Error> {
        let mut state = UpgradeState {
            padding: Default::default(),
            module: self,
            upgraded_handles: Default::default(),
            unresolved_handles: Default::default(),
            needle_and_replacement: None,
        };

        for op in ops.into_iter() {
            let padding = state.inc_padding();
            padding.debug("op: ", op);

            // Find the expression's enclosing function, if any
            let mut maybe_fn_handle = None;
            for (fn_handle, function) in state.module.functions.iter() {
                log::trace!("function: {fn_handle:?}");
                if function.expressions.try_get(op.pointer_handle).is_ok() {
                    log::trace!("  is op's function");
                    maybe_fn_handle = Some(fn_handle);
                    break;
                }
            }

            // Upgrade the pointer's type and then manually add that type to the list of
            // unresolved upgrades.
            // We do this because we want to search for references to this type, but types *are not*
            // automatically added to unresolved upgrades.
            // For more info see [`UpgradeState::upgrade_type`].
            padding.debug("upgrading the pointer type:", op.pointer_type_handle);
            let new_pointer_type_handle = state.upgrade_type(op.pointer_type_handle)?;
            state.add_upgraded(op.pointer_type_handle, new_pointer_type_handle);

            // Upgrade the pointer's expression. The upgrade will be automatically recorded.
            padding.debug("upgrading the pointer expression", op.pointer_handle);
            let _new_pointer_handle =
                state.upgrade_expression(maybe_fn_handle, op.pointer_handle)?;
        }
        log::trace!("all upgraded: {:#?}", state.upgraded_handles);

        // Run through all unresolved upgrades and search for references to them in other things,
        // spiraling outward until there are no more unresolved upgrades.
        while let Some(nr @ (needle, replacement)) = state.pop_unresolved() {
            state.padding.debug("searching for: ", needle);
            state.padding.debug("replacing with: ", replacement);
            state.needle_and_replacement = Some(nr);
            let all_haystacks = state.get_haystacks(needle);
            for haystack in all_haystacks {
                state.search(haystack)?;
            }
        }

        Ok(())
    }
}
